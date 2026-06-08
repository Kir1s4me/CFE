import logging
import os
import time
from datetime import datetime

import torch
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from models.model_utils import PCViews, fps_subsample
from models.CFE import Model
import open3d as o3d


def _get_core(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _count_params(model: torch.nn.Module):
    core = _get_core(model)
    total = sum(p.numel() for p in core.parameters())
    trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
    return total, trainable


def _bytes_to_mb(b: int) -> float:
    return b / (1024 ** 2)


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None, save_results=True):
    torch.backends.cudnn.benchmark = True

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()
    t_total_start = time.perf_counter()

    if save_results:
        save_dir = os.path.join(cfg.DIR.OUT_PATH, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"All results will be saved to: {save_dir}")
    else:
        save_dir = None

    if test_data_loader is None:
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TEST),
            batch_size=1,
            num_workers=4,
            collate_fn=utils.data_loaders.collate_fn_55,
            pin_memory=True,
            shuffle=False
        )

    if model is None:
        model = Model(cfg)
        if use_cuda:
            model = torch.nn.DataParallel(model).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location="cpu")
        model.load_state_dict(checkpoint['model'])

    model.eval()


    total_p, trainable_p = _count_params(model)
    print('=========================== MODEL PROFILE ===========================')
    print(f'Total parameters    : {total_p:,} ({total_p/1e6:.3f} M)')
    print(f'Trainable parameters: {trainable_p:,} ({trainable_p/1e6:.3f} M)')
    print('=====================================================================')

    peak_alloc_mb = 0.0
    peak_resv_mb = 0.0
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


    n_samples = len(test_data_loader)
    test_metrics = AverageMeter(['CD', 'DCD', 'F1'])
    category_metrics = dict()
    mclass_metrics = AverageMeter(['CD', 'DCD', 'F1'])
    render = PCViews(TRANS=-cfg.NETWORK.view_distance, RESOLUTION=224)

    crop_ratio = {'easy': 1/4, 'median': 1/2, 'hard': 3/4}
    choice = [
        torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]), torch.Tensor([1, -1, 1]), torch.Tensor([-1, 1, 1]),
        torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]), torch.Tensor([1, -1, -1]), torch.Tensor([-1, -1, -1])
    ]

    mode = cfg.CONST.mode
    print('Start evaluating (mode: {:s}) ...'.format(mode))

    with tqdm(test_data_loader) as t:
        for batch_idx, (taxonomy_id, model_ids, data) in enumerate(t):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                gt = data['gtcloud']
                gt_ = gt[:, :, :3].contiguous()
                _, npoints, _ = gt.size()


                if save_results:
                    sample_dir = f"sample_{batch_idx:04d}_{model_ids[0]}"
                    sample_path = os.path.join(save_dir, sample_dir)
                    os.makedirs(sample_path, exist_ok=True)


                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(gt_[0].cpu().numpy())
                    o3d.io.write_point_cloud(os.path.join(sample_path, "gt.ply"), pcd)

                num_crop = int(npoints * crop_ratio[mode])

                for partial_id, item in enumerate(choice):
                    partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, num_crop, fixed_points=item)
                    partial = fps_subsample(partial, 2048)
                    partial_ = partial[:, :, :3].contiguous()

                    if save_results:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(partial_[0].cpu().numpy())
                        o3d.io.write_point_cloud(
                            os.path.join(sample_path, f"view_{partial_id:02d}_partial.ply"),
                            pcd
                        )

                    #
                    pcds_pred = model(partial.contiguous())

                    if save_results:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pcds_pred[-1][0].cpu().numpy())
                        o3d.io.write_point_cloud(
                            os.path.join(sample_path, f"view_{partial_id:02d}_pred.ply"),
                            pcd
                        )

                    #
                    cdl1, cdl2, f1 = calc_cd(pcds_pred[-1], gt_, calc_f1=True)
                    dcd, _, _ = calc_dcd(pcds_pred[-1], gt_)

                    test_metrics.update([cdl2.mean().item() * 1e3, dcd.mean().item(), f1.mean().item()])
                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(['CD', 'DCD', 'F1'])
                    category_metrics[taxonomy_id].update([cdl2.mean().item() * 1e3, dcd.mean().item(), f1.mean().item()])


                    if use_cuda:
                        torch.cuda.synchronize()
                        peak_alloc_mb = max(peak_alloc_mb, _bytes_to_mb(torch.cuda.max_memory_allocated()))
                        peak_resv_mb = max(peak_resv_mb, _bytes_to_mb(torch.cuda.max_memory_reserved()))

                t.set_description(
                    'Test[%d/%d]  Average Metrics = %s' %
                    (batch_idx, n_samples, ['%.4f' % l for l in test_metrics.avg()])
                )

    # 输出测试结果
    print('============================ TEST RESULTS ============================')
    print('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

    for taxonomy_id in category_metrics:
        message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0))
        message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
        mclass_metrics.update(category_metrics[taxonomy_id].avg())
        print(message)

    print('Overall\t{:d}\t'.format(test_metrics.count(0)) + '\t'.join(['%.4f' % value for value in test_metrics.avg()]))
    print('MeanClass\t\t' + '\t'.join(['%.4f' % value for value in mclass_metrics.avg()]))

    print('=========================== GPU MEMORY =============================')
    if use_cuda:
        if peak_alloc_mb == 0.0 and peak_resv_mb == 0.0:
            torch.cuda.synchronize()
            peak_alloc_mb = _bytes_to_mb(torch.cuda.max_memory_allocated())
            peak_resv_mb = _bytes_to_mb(torch.cuda.max_memory_reserved())
        print(f'Peak max_memory_allocated: {peak_alloc_mb:.2f} MB')
        print(f'Peak max_memory_reserved : {peak_resv_mb:.2f} MB')
    else:
        print('CUDA not available. Peak GPU memory usage is not measured.')

    if use_cuda:
        torch.cuda.synchronize()
    t_total_end = time.perf_counter()
    total_sec = t_total_end - t_total_start

    print('=========================== TOTAL TIME ==============================')
    print(f'Total inference time: {total_sec:.3f} s')
    print('=====================================================================')