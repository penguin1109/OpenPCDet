import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBEVBackbone(nn.Module):
  def __init__(self, model_cfg, input_channels):
    super(UNetBEVBackbone, self).__init__()
    self.model_cfg = model_cfg
    self.input_channels = input_channels
    assert len(model_cfg.UPSAMPLE_FILTERS) == len(model_cfg.DOWNSAMPLE_FILTERS)
    assert model_cfg.LAYER_NUMS[0] == len(model_cfg.DOWNSAMPLE_FILTERS)
    assert model_cfg.LAYER_NUMS[1] == len(model_cfg.UPSAMPLE_FILTERS)

    self.down_blocks = []
    self.up_blocks = []

  
    for i in range(len(model_cfg.DOWNSAMPLE_FILTERS)):
      self.down_blocks.append(nn.Sequential(
          nn.MaxPool2d(kernel_size=model_cfg.LAYER_STRIDES[0], padding=0),
          nn.Conv2d(input_channels, model_cfg.DOWNSAMPLE_FILTERS[i], kernel_size = 3, stride = 1, padding=1),
          nn.BatchNorm2d(model_cfg.DOWNSAMPLE_FILTERS[i], eps=1e-3, momentum=0.01), nn.ReLU(),
          nn.Conv2d(model_cfg.DOWNSAMPLE_FILTERS[i], model_cfg.DOWNSAMPLE_FILTERS[i], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(model_cfg.DOWNSAMPLE_FILTERS[i], eps =1e-3, momentum=0.01), nn.ReLU()
      ))
      input_channels = model_cfg.DOWNSAMPLE_FILTERS[i]
    
    self.upsample = [
        nn.ConvTranspose2d(model_cfg.UPSAMPLE_FILTERS[0], model_cfg.UPSAMPLE_FILTERS[1], kernel_size = 2, padding=0, stride = 2, output_padding=0),
        nn.ConvTranspose2d(model_cfg.UPSAMPLE_FILTERS[1], self.input_channels, kernel_size = 2, padding=0, stride =2, output_padding=0)]
    
  
    for i in range(len(model_cfg.UPSAMPLE_FILTERS)-1):
      self.up_blocks.append(nn.Sequential(
          nn.Conv2d(model_cfg.UPSAMPLE_FILTERS[i], model_cfg.UPSAMPLE_FILTERS[i+1], kernel_size = 3, padding = 1, stride = 1),
          nn.BatchNorm2d(model_cfg.UPSAMPLE_FILTERS[i+1], eps=1e-3, momentum=0.01), nn.ReLU(),
          nn.Conv2d(model_cfg.UPSAMPLE_FILTERS[i+1],model_cfg.UPSAMPLE_FILTERS[i+1], kernel_size = 3, padding = 1, stride = 1),
          nn.BatchNorm2d(model_cfg.UPSAMPLE_FILTERS[i+1], eps=1e-3, momentum=0.01), nn.ReLU()
      ))
    self.up_blocks.append(nn.Sequential(
          nn.Conv2d(model_cfg.UPSAMPLE_FILTERS[-1], model_cfg.UPSAMPLE_FILTERS[-1], kernel_size = 1, padding = 0, stride = 1),
          nn.BatchNorm2d(model_cfg.UPSAMPLE_FILTERS[-1], eps=1e-3, momentum=0.01), nn.ReLU(),
    ))
    self.num_bev_features = self.model_cfg.UPSAMPLE_FILTERS[-1]


  
  def forward(self, data_dict, **kwargs):
    spatial_features = data_dict['spatial_features'] # [B, 256, H, W]
    concat_blocks = []
    concat_blocks.append(spatial_features)
    for idx, block in enumerate(self.down_blocks):
      spatial_features = block(spatial_features)
      #print(spatial_features.shape)
      if idx == len(self.down_blocks)-1:
        continue
      concat_blocks.append(spatial_features)

    concat_blocks = concat_blocks[::-1]
    
    for idx, block in enumerate(self.up_blocks):
      spatial_features = self.upsample[idx](spatial_features)
      print(spatial_features.shape)
      spatial_features = torch.cat([concat_blocks[idx], spatial_features], dim=1)
      spatial_features = block(spatial_features)
    
    data_dict['spatial_features_2d'] = spatial_features

    return data_dict
    