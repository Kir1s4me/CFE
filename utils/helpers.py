

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
import random
from models.model_utils import fps_subsample
import torch.nn.functional as F


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d or \
       type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def count_parameters(network):
    return sum(p.numel() for p in network.parameters())


def seprate_point_cloud(xyz_features,
                        num_points,
                        crop,
                        fixed_points=None,
                        padding_zeros=False):
    '''
    Separate point cloud with features: usage: generate incomplete point cloud with specified number of points.
    Input shape: [B, N, 7] where first 3 columns are xyz, last 4 columns are features
    '''
    batch_size, n, c = xyz_features.shape

    assert n == num_points
    assert c == 7
    if crop == num_points:
        return xyz_features, None

    INPUT = []
    CROP = []
    
    for points in xyz_features:
        if isinstance(crop, list):
            num_crop = random.randint(crop[0], crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)  # [1, N, 7]

        #
        xyz_part = points[:, :, :3]  # [1, N, 3]
        features_part = points[:, :, 3:]  # [1, N, 4]

        if fixed_points is None:
            center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1).cuda()
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3).cuda()

        #
        distance_matrix = torch.norm(center.unsqueeze(2) - xyz_part.unsqueeze(1),
                                     p=2,
                                     dim=-1)  # [1, 1, N]

        idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0, 0]  # [N]

        if padding_zeros:
            input_data = points.clone()  # [1, N, 7]
            #
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0
            crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)  # [1, M, 7]
        else:
            #
            input_data = torch.cat([
                xyz_part[0, idx[num_crop:]].unsqueeze(0),  # [1, N-M, 3]
                features_part[0, idx[num_crop:]].unsqueeze(0)  # [1, N-M, 4]
            ], dim=-1)  # [1, N-M, 7]
            
            crop_data = torch.cat([
                xyz_part[0, idx[:num_crop]].unsqueeze(0),  # [1, M, 3]
                features_part[0, idx[:num_crop]].unsqueeze(0)  # [1, M, 4]
            ], dim=-1)  # [1, M, 7]

        if isinstance(crop, list):

            input_data_sampled = fps_subsample(input_data, 2048)

            crop_data_sampled = fps_subsample(crop_data, 2048)
            INPUT.append(input_data_sampled)
            CROP.append(crop_data_sampled)
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT, dim=0)  # [B, N', 7]
    crop_data = torch.cat(CROP, dim=0)  # [B, M, 7]

    return input_data.contiguous(), crop_data.contiguous()