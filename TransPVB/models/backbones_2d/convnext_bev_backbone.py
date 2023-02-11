import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtModule(nn.Module):
  def __init__(self, input_channels, hidden_channels):
    super(ConvNeXtModule, self).__init__()
    self.layers = []
    for i in range(3):
      self.layers.append(nn.Sequential(
        nn.Conv2d(input_channels, input_channels, kernel_size=7, groups=input_channels, stride=1, padding=3),
        nn.LayerNorm2d(input_channels), nn.Conv2d(input_channels, hidden_channels, kernel_size=1), nn.GELU(),
        nn.Conv2d(hidden_channels, input_channels, kernel_size=1)))
  
  def forward(self, x):
    for i in range(len(self.layers)):
      x = self.layers[i](x) + x
    return x

class ConvNeXtBEVBackbone(nn.Module):
  def __init__(self, model_cfg, input_channels, **kwargs):
    super(ConvNeXtBEVBackbone, self).__init__()
    self.model_cfg = model_cfg
    self.input_channels = input_channels
    self.hidden_channels = model_cfg.HIDDEN_CHANNELS
    self.layer1 = nn.Sequential(
        ConvNeXtModule(input_channels, self.hidden_channels[0]),
        nn.Conv2d(input_channels, input_channels//4, kernel_size=3, padding=1, stride=1)
    )
    self.layer2 = nn.Sequential(
        ConvNeXtModule(input_channels // 4, self.hidden_channels[1]),
        nn.Conv2d(input_channels//4, input_channels//2, kernel_size=3, padding=1, stride=1)
    )
    self.deconv1 = nn.ConvTranspose2d(input_channels//4, input_channels//2, kernel_size=1, padding=0, output_padding=0)
    self.deconv2 = nn.ConvTranspose2d(input_channels//2, input_channels//2, kernel_size=3, padding=1, output_padding=1)

    self.num_bev_features = input_channels
  
  def forward(self, data_dict, **kwargs):
    spatial_features = data_dict['spatial_features'] # [B, 256, H, W]
    out1 = self.layer1(spatial_features)
    dconv1 = self.deconv1(out1)
    out2 = self.layer2(out1)
    dconv2 = self.deconv2(out2)

    out = torch.cat((dconv1, dconv2), dim=1)
    data_dict['spatial_features_2d'] = out

    return out