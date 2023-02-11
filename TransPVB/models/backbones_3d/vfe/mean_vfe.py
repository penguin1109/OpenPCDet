## Mean Voxel Encoder

import torch
import torch.nn as nn

from pcdet.models.backbones_3d.vfe.vfe_template import VFETemplate

class MeanVoxelFeatureEncoder(VFETemplate):
  def __init__(self, model_cfg, num_point_features, **kwargs):
    super().__init__(model_cfg = model_cfg)
    self.num_point_features = num_point_features
  
  def get_output_feature_dim(self):
    return self.num_point_features
  
  def forward(self, batch_dict, **kwargs):
    voxel_features = batch_dict['voxels'] ## [#of non empty voxel, 5, # point features] -> 우선 최대 5개인데 voxel_num_points를 사용해서 몇개 사용하는지. 
    voxel_num_points = batch_dict['voxel_num_points'] ## 각각의 voxel에 몇개의 point들이 포함되어 있는지 (최대 5개라는데..ㅎ)
    ## MY CODE -> BUT WAS WRONG ##
    # mean_vox_features = torch.mean(voxel_features, dim=1) ## 마지막 차원에 대해서 평균을 내어 준다. -> but 단순 이렇게 평균을 내어 주면 point개수가 5개가 아니면? -> 이러면 평균을 잘못 구하는 상황임임
    #voxel_features = torch.Tensor(voxel_features)
    #voxel_num_points= torch.Tensor(voxel_num_points)
    mean_vox_features = voxel_features[:, :, :].sum(dim =1, keepdim=False) ## 이렇게 하면 voxel의 max 개의 point들에 대해서 feature을 더해 준다.
    normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), 1.0).type_as(voxel_features) ## voxel내에 point가 0개인것의 경우에는
    mean_vox_features /= normalizer
    batch_dict['voxel_features'] = mean_vox_features.contiguous()
    return batch_dict
