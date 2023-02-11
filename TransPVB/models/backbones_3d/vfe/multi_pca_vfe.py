## Mutli Scale Point Channel Attention
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.models.backbones_3d.vfe.vfe_template import VFETemplate

class PAModule(nn.Module):
  def __init__(self, initial_features, hidden_features):
    super(PAModule, self).__init__()
    self.in_features = initial_features # [K, N, C]
    self.layer = nn.Sequential(
        nn.Linear(self.in_features, hidden_features, bias=False), nn.ReLU(), nn.Linear(hidden_features, self.in_features, bias=False)
    )
 

  def forward(self, x):
    max_x = torch.max(x, dim=-1, keepdim=False)
    avg_x = torch.mean(x, dim=-1, keepdim=False)
    max_x = self.layer(max_x)
    avg_x = self.layer(avg_x)

    out_x = max_x + avg_x
    out_x = F.sigmoid(out_x)
    return out_x

class CAModule(nn.Module):
  def __init__(self, initial_features, hidden_features):
    super(CAModule, self).__init__()
    self.layer = nn.Sequential(
        nn.Linear(initial_features, hidden_features, bias=False), nn.ReLU(), nn.Linear(hidden_features, initial_features, bias=False)
    )
  
  def forward(self, x):
    max_x = torch.max(x, dim=1, keepdim=False)
    avg_x = torch.mean(x, dim=1, keepdim=False)
    max_x = self.layer(max_x)
    avg_x = self.layer(avg_x)

    out_x = max_x + avg_x
    out_x = F.sigmoid(out_x)
    return out_x

class MultiPCAVoxelFeatureEncoder(VFETemplate):
  def __init__(self, model_cfg, num_point_features, **kwargs):
    super().__init__(model_cfg = model_cfg)
    self.num_point_features = num_point_features
    self.middle_features = model_cfg.get('MIDDLE_CH', None)
    if self.middle_features is None:
      self.middle_features = 32

    self.pa1 = PAModule(num_point_features, self.middle_features)
    self.pa2 = PAModule(num_point_features*2, self.middle_features)
    self.ca1 = CAModule(num_point_features, self.middle_features)
    self.ca2 = CAModule(num_point_features*2, self.middle_features)

    self.fc = nn.Linear(self.middle_features, self.middle_features)
  
  def get_output_feature_dim(self):
    return self.num_point_features
  
  def forward(self, batch_dict, **kwargs):
    voxel_features = batch_dict['voxels'] # [K, N, C]
    x1, x2 = self.pa1(voxel_features), self.ca1(voxel_features)
    out = voxel_features * x1 * x2
    voxel_features = torch.cat((out, voxel_features), dim=1)
    x1, x2 = self.pa2(voxel_features), self.ca2(voxel_features)
    out = voxel_features * x1 * x2
    voxel_features = out + voxel_features

    voxel_features = torch.max(self.fc(voxel_features), dim=1) # [K, N, C]
    batch_dict['voxel_features'] = voxel_features

    return batch_dict

