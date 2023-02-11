import torch.nn as nn
!pip install einops
from einops import rearrange

class HeightCompression(nn.Module):
  def __init__(self, model_cfg, **kwargs):
    super(HeightCompression, self).__init__()
    self.model_cfg = model_cfg
    self.bev_features = self.model_cfg['NUM_BEV_FEATURES']
  
  def forward(self, batch_dict):
    encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
    spatial_features  = encoded_spconv_tensor.dense()
    B, C, D, H, W = spatial_features.shape
    spatial_features = rearrange(spatial_features, 'b c d h w -> b (c d) h w') ## c = 128, d(=depth) = 2이기 때문에 결국에 BEV feature의 channel dimension은 128x2 = 256이다.
    batch_dict['spatial_features'] = spatial_features
    batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

    return batch_dict
