from ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils

import torch
import torch.nn as nn

""" MySetAggregation
- PoinrNet2Utils에서 bulid_local_aggreataion_module을 대신해서 사용하고자 한다.
"""
class MySetAggregation(nn.Module):
  def __init__(self, use_xyz,  config, mlps):
    super(MySetAggregation, self).__init__()
    self.config = config
    self.radii = self.config.POOL_RADIUS
    self.nsamples = self.config.NSAMPLE
    self.mlps_channels = mlps # self.config.MLPS

    assert len(self.radii) == len(self.nsamples) == len(self.mlps_channels)
    # print(mlps)
    self.groupers = nn.ModuleList()
    self.mlps = nn.ModuleList()
    for i in range(len(self.radii)):
      radius = self.radii[i]
      nsample = self.nsamples[i] # 원래 논문에서는 random sampling을 한다고 했었는데, 따로 샘플링을 하는 줄 알았더니 그냥 랜덤하게 고르는 것이었다.() 애초부터 그냥 정해진 point sample의 개수만 고른다.)
      self.groupers.append(pointnet2_stack_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
      mlp_spec = self.mlps_channels[i]
      if use_xyz:
        mlp_spec[0] += 3
      
      shared_mlps = []
      for k in range(len(mlp_spec)-1):
        shared_mlps.extend([
            nn.Conv2d(mlp_spec[k], mlp_spec[k+1], kernel_size=1, bias=False),
            nn.BatchNorm2d(mlp_spec[k+1]), nn.ReLU()
        ])
      self.mlps.append(nn.Sequential(*shared_mlps))
    self.pool_method = 'max_pool'
  
    self.init_weights()
  
  def init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

  def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
    new_features_list = []
    for k in range(len(self.groupers)):
      new_features, ball_idxs = self.groupers[k](
          xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
      ) # (M1 + M2, C, nsample)
      new_features = new_features.permute(1,0,2).unsqueeze(dim=0) # (1, C, M1+M2, nsample)
      new_features = self.mlps[k](new_features) # (1, c, M1+M2, nsample)

      new_features = F.max_pool2d(new_features, kernel_size = [1, new_features.size(3)]) # kernel_size = [1, nsample]
      new_features = new_features.squeeze(dim=-1) # (1, C, M1+M2)
      new_features = new_features.squeeze(dim=0).permute(1, 0) # (M1+M2, C)
      new_features_list.append(new_features)
    new_features = torch.cat(new_features_list, dim=1) # (M1+M2 .., C)

    return new_xyz, new_features