## Inverse Distance Voxel Feature Encoder (w. Shallow Point Net for Segmentation)

import torch
import torch.nn as nn
import math

from pcdet.models.backbones_3d.vfe.vfe_template import VFETemplate

class InverseDistanceVoxelFeatureEncoder(VFETemplate):
  def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
    super().__init__(model_cfg=model_cfg)
    # 여기서 num_point_features: 각각의 raw point를 (x,y,z)로만 나타내는지, 아니면 다른 정보도 사용하는지
    self.num_point_features = num_point_features
    self.voxel_size = voxel_size
    self.point_cloud_range = point_cloud_range

  def get_output_feature_dim(self):
    return self.num_point_features

  def _make_voxel_center(self):
    x_offsets = torch.arange(self.point_cloud_range[0], self.point_cloud_range[3], step=self.voxel_size[0]) + (self.voxel_size[0] / 2)
    y_offsets = torch.arange(self.point_cloud_range[1], self.point_cloud_range[4], step=self.voxel_size[1]) + (self.voxel_size[1] / 2)
    z_offsets = torch.arange(self.point_cloud_range[2], self.point_cloud_range[5], step=self.voxel_size[2]) + (self.voxel_size[2] / 2)
    x,y,z = torch.meshgrid([
        x_offsets, y_offsets, z_offsets
    ])
    center_points = torch.stack((x, y, z), dim=-1)
    return center_points
  
  def _weight_mask(self, cx, cy, cz, voxel, npts):
    normalize_term = 0.0
    new_voxel = torch.zeros(voxel.shape).to(voxel.device)

    for i in range(npts.type(torch.int).item()):
      px, py, pz, occ = voxel[i] # voxel에 들어 있는 point의 3D coordinate와 occlusion 정보
      dist = abs(px-cx) + abs(py-cy) + abs(pz-cz)
      #print(dist, math.pow(dist, -1))
      normalize_term += math.pow(dist, -1)
      new_voxel[i] += math.pow(dist, -1) * voxel[i]
    
    normalize_term = torch.tensor(normalize_term)
    #print(new_voxel)
    mean_feature = new_voxel[:,:].sum(dim=0, keepdim=False)
    # normalize_term = torch.clamp_min(normalize_term, 1.0).type_as(mean_feature)
    if normalize_term == torch.tensor(0.0):
      normlize_term = torch.tensor(1.0)
    mean_feature /= normalize_term
    mean_feature /= torch.clamp_min(npts,1.0).type_as(mean_feature) ## [,4]
    mean_features = torch.unsqueeze(mean_feature, dim=0)

    return mean_feature

  def _make_voxel_feature(self, voxels, voxel_num_points, voxel_coords):
    # 만약에 point중에서 voxel의 중심에 해당하는 point가 있다면 어떻게 하지?
    # 게다가 point가 1개밖에 없다면 어떻게 하는게 좋을까?
    # (1) voxel들의 중점을 우선 구해 준다.
    center_points = self._make_voxel_center().to(voxels.device)
    new_voxel_features = []
    # (2) voxel 순서대로 center point과의 거리로 나타내어 준다.
    for voxel, npts, coord in zip(voxels, voxel_num_points, voxel_coords):
      coord = coord.type(torch.int)
      cx, cy, cz = center_points[coord[3].item(), coord[2].item(), coord[1].item()]
      new_voxel_features.append(self._weight_mask(cx, cy, cz, voxel, npts))
    # (3) 첫번쨰 차원, 즉 non-empty voxel의 개수 차원에 대해서 stacking을 해서 최종 feature vector을 구한다.
    new_voxel_features = torch.stack(new_voxel_features, dim=0)
    return new_voxel_features

  def forward(self, batch_dict, **kwargs):
    voxels = batch_dict['voxels']
    voxel_num_points = batch_dict['voxel_num_points']
    voxel_coords = batch_dict['voxel_coords']
    voxel_features = self._make_voxel_feature(voxels, voxel_num_points, voxel_coords)
    batch_dict['voxel_features'] = voxel_features

    return batch_dict



