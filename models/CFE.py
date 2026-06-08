from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from pointnet2_ops.pointnet2_utils import gather_operation as gather_points
import time
from models.model_utils import *
from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from models.dgcnn_utils import *


class ENCONet(nn.Module):
    def __init__(self, cfg):
        super(ENCONet, self).__init__()
        self.channel = 64
        self.view_distance = cfg.NETWORK.view_distance
        self.relu = nn.GELU()
        self.sa = self_attention(self.channel * 8, self.channel * 8, dropout=0.0)
        self.viewattn = self_attention(256, 256)

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(512 + self.channel * 4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(512, self.channel, 128, bias=True)

        self.ps2 = nn.ConvTranspose1d(64, 64, kernel_size=4, stride=4, bias=True)

        self.ps_refuse = nn.Conv1d(512 + self.channel, self.channel * 8, kernel_size=1)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 256)
        )
        self.inc_dim = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=1)
        )
        self.dgcnn = DGCNN_Grouper()

    def forward(self, points):
        batch_size, _, N = points.size()
        points = points[:, :4, :]
        f_p, point_ = self.dgcnn(points)  # b 128(c) 256(n)
        # print("fp",f_p.shape)

        f_p = self.inc_dim(f_p)

        pos = self.pos_embed(point_.permute(0, 2, 1).contiguous())  # b c=128 256(n)

        f_p_ = F.adaptive_max_pool1d(f_p, 1)

        f_v_ = self.viewattn(f_p + pos.permute(0, 2, 1).contiguous())  # no emb

        f_v_ = F.adaptive_max_pool1d(f_v_, 1)
        f_g = torch.cat([f_p_, f_v_], 1)  # b 512(c) 1

        x = self.relu(self.ps(f_g))  # b 64(c) 128

        x = self.relu(self.ps2(x))  # b 64(c) 512
        x = self.relu(self.ps_refuse(torch.cat([x, f_g.repeat(1, 1, x.size(2))], 1)))  # b 512(c) 512
        # print("x", x.shape)
        x2_d = (self.sa(x)).reshape(batch_size, self.channel * 4, N // 2)  # b 256(c) 512

        coarse = self.conv_out(self.relu(self.conv_out1(torch.cat([x2_d, f_g.repeat(1, 1, x2_d.size(2))], 1))))
        return f_g, coarse

class SDG(nn.Module):
    def __init__(self, channel=128,ratio=1,hidden_dim = 512,dataset='ShapeNet'):
        super(SDG, self).__init__()
        self.channel = channel
        self.hidden = hidden_dim

        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = self_attention(channel*2,hidden_dim,dropout=0.0,nhead=8)
        self.cross1 = cross_attention(hidden_dim, hidden_dim, dropout=0.0,nhead=8)

        self.decoder1 = SDG_Decoder(hidden_dim,channel,ratio) if dataset == 'ShapeNet' else self_attention(hidden_dim, channel * ratio, dropout=0.0,nhead=8)

        self.decoder2 = SDG_Decoder(hidden_dim,channel,ratio) if dataset == 'ShapeNet' else self_attention(hidden_dim, channel * ratio, dropout=0.0,nhead=8)

        self.relu = nn.GELU()
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_delta = nn.Conv1d(channel, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel*ratio*2, channel*ratio, kernel_size=1)
        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.mlpp = MLP_CONV(in_channel=256,layer_dims=[256,hidden_dim])
        self.sigma = 0.2
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.cd_distance = chamfer_3DDist()
        self.pos_embed = nn.Sequential(
                    nn.Linear(3, 128),
                    nn.GELU(),
                    nn.Linear(128, 256),
                    nn.GELU(),
                    nn.Linear(256, hidden_dim),
                )

    def forward(self, local_feat, coarse,f_g,partial):

        batch_size, _, N = coarse.size()
        F_ = self.conv_x1(self.relu(self.conv_x(coarse)))
        f_g = self.conv_1(self.relu(self.conv_11(f_g)))
        F_ = torch.cat([F_, f_g.repeat(1, 1, F_.shape[-1])], dim=1)
        

        # Structure Analysis
        _, _, n2 = F_.size()
        point_ = partial.permute(0, 2, 1) # b n 3

        pos = self.pos_embed(point_) # b n=2048 256
        pos = pos.permute(0, 2, 1)

        pos = F.adaptive_max_pool1d(pos, n2).permute(2, 0, 1) # b 256 3

        F_Q = self.sa1(F_,pos)
        F_Q_ = self.decoder1(F_Q)

        # Similarity Alignment
        local_feat = self.mlpp(local_feat)
        F_H = self.cross1(F_Q,local_feat)
        F_H_ = self.decoder2(F_H)

        F_L = self.conv_delta(self.conv_ps(torch.cat([F_Q_,F_H_],1)).reshape(batch_size,-1,N*self.ratio))
        O_L = self.conv_out(self.relu(self.conv_out1(F_L)))
        fine = coarse.repeat(1,1,self.ratio) + O_L

        return fine

class local_encoder(nn.Module):
    def __init__(self,cfg):
        super(local_encoder, self).__init__()
        self.gcn_1 = EdgeConv(3, 64, 16)
        self.gcn_2 = EdgeConv(64, 256, 8)
        self.local_number = cfg.NETWORK.local_points

    def forward(self,input):
        x1 = self.gcn_1(input)
        idx = furthest_point_sample(input.transpose(1, 2).contiguous(), self.local_number)
        x1 = gather_points(x1,idx)
        x2 = self.gcn_2(x1)

        return x2

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()

        self.encoder = ENCONet(cfg)
        self.localencoder = local_encoder(cfg)
        self.merge_points = cfg.NETWORK.merge_points
        self.refine1 = SDG(ratio=cfg.NETWORK.step1,hidden_dim=768,dataset=cfg.DATASET.TEST_DATASET)
        self.refine2 = SDG(ratio=cfg.NETWORK.step2,hidden_dim=512,dataset=cfg.DATASET.TEST_DATASET)

        self.distance_threshold = 0.05 
        self.target_points = 2048 #
    def find_missing_points_advanced(self, partial, sparse_pc):
        """

        """
        batch_size = partial.shape[0]
        device = partial.device
        
        missing_points_unified = torch.zeros((batch_size, 3, self.target_points), device=device)
        
        for i in range(batch_size):
            dist_matrix = torch.cdist(sparse_pc[i], partial[i])
            min_distances, _ = torch.min(dist_matrix, dim=1)
            
            missing_mask = min_distances > self.distance_threshold
            missing_indices = torch.where(missing_mask)[0]
            
            if len(missing_indices) > 0:
                missing_points = sparse_pc[i][missing_indices]  # (k, 3)
                
                if len(missing_indices) >= self.target_points:
                    #
                    sampled_indices = furthest_point_sample(
                        missing_points.unsqueeze(0).transpose(1, 2).contiguous(), 
                        self.target_points
                    )[0]
                    sampled_indices = sampled_indices.to(torch.long)
                    selected_points = missing_points[sampled_indices].transpose(0, 1)
                    missing_points_unified[i] = selected_points
                else:
                    #
                    missing_points = missing_points.transpose(0, 1)  # (3, k)
                    repeat_times = self.target_points // len(missing_indices)
                    remainder = self.target_points % len(missing_indices)
                    
                    repeated = missing_points.repeat(1, repeat_times)
                    if remainder > 0:
                        remainder_points = missing_points[:, :remainder]
                        missing_points_unified[i] = torch.cat([repeated, remainder_points], dim=1)
                    else:
                        missing_points_unified[i] = repeated
            else:
                #
                missing_points_unified[i] = torch.zeros((3, self.target_points), device=device)
        
        return missing_points_unified



    def forward(self, partial):
        partial = partial.transpose(1,2).contiguous()
        feat_g, coarse = self.encoder(partial)
        
        # print("coarse", coarse.shape)
        part_3 = partial[:, :3, :]    

        local_feat = self.localencoder(part_3)  
        coarse_merge = torch.cat([part_3,coarse],dim=2)
        coarse_merge = gather_points(coarse_merge, furthest_point_sample(coarse_merge.transpose(1, 2).contiguous(), self.merge_points))


        fine1 = self.refine1(local_feat, coarse_merge, feat_g,part_3)

        fine1_transposed = fine1.transpose(1, 2).contiguous()  # (b, n, 3)
        partial_transposed = part_3.transpose(1, 2).contiguous()  # (b, n, 3)
        missing_points= self.find_missing_points_advanced(partial_transposed, fine1_transposed)

        fine2 = self.refine2(local_feat, fine1, feat_g,missing_points)

        return (coarse.transpose(1, 2).contiguous(),fine1.transpose(1, 2).contiguous(),fine2.transpose(1, 2).contiguous())




