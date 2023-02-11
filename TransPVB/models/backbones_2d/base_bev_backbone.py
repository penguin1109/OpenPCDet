import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseBEVBackbone(nn.Module):
  def __init__(self, model_cfg, input_channels):
    super(BaseBEVBackbone, self).__init__()
    self.model_cfg = model_cfg

    if self.model_cfg.get('LAYER_NUMS', None) is not None:
      assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
      layer_nums = self.model_cfg.LAYER_NUMS # [5,5]
      layer_strides = self.model_cfg.LAYER_STRIDES # [1,2] -> 유지하고 downsample 한번
      num_filters = self.model_cfg.NUM_FILTERS # [128, 128]
    else:
      layer_nums = layer_strides = num_filters = []
  
    if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
      assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
      upsample_strides = self.model_cfg.UPSAMPLE_STRIDES # [1, 2] -> upsample 한번
      num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS # [256, 256]
    else:
      upsample_strides = num_upsample_filters = []
    
    num_levels = len(layer_nums)
    c_in_list = [input_channels, *num_filters[:-1]]
    self.blocks = nn.ModuleList()
    self.deblocks = nn.ModuleList()

    for idx in range(num_levels): ## 2 (처음에는 spatial size의 변화가 없다가 다음에는 2배로 downsample하고 이어서 2배로 upsample 한다.)
      cur_layers = [
          nn.ZeroPad2d(1),
          nn.Conv2d(
              c_in_list[idx], num_filters[idx], kernel_size=3, stride = layer_strides[idx], padding=0, bias=False  ## 처음만 stride가 1이 아닌 경우가 있고고 나머지 layer들은 stride가 무조건 1이 다.
          ),
          nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
          nn.ReLU()
      ]
      for k in range(layer_nums[idx]): ## 5, 5
        cur_layers.extend([
            nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
            nn.ReLU()
        ]) ## 계속 같은 128개의 channel dimension을 가지게 됨을 확인할 수 있다.

      self.blocks.append(nn.Sequential(*cur_layers))

      if len(upsample_strides) > 0:
        stride = upsample_strides[idx]
        if stride >= 1:
          self.deblocks.append(nn.Sequential(
              nn.ConvTranspose2d(
                  num_filters[idx], num_upsample_filters[idx], upsample_strides[idx], stride=upsample_strides[idx], bias=False
              ),
              nn.BatchNorm2d(num_upsample_filters[idx], eps = 1e-3, momentum=0.01), nn.ReLU()
          ))
        else:
          stride = np.round(1/stride).astype(np.int)
          self.deblocks.append(nn.Sequential(
              nn.Conv2d(num_filters[idx], num_upsample_filters[idx], stride, stride=stride, bias=False),
              nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01), nn.ReLU()
          ))

    c_in = sum(num_upsample_filters) ## sum([256, 256]) = 512
    if len(upsample_strides) > num_levels:
      self.deblocks.append(nn.Sequential(
          nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
          nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01), nn.ReLU()
      ))
    self.num_bev_features = c_in ## 여기서 bev feature의 2D Backbone을 거친 output의 channel dimension의 결과가 나온다. -> 따라서 여기서는 512로 concat을 하였음에도 불구하고 다른 2D Backbone을 사용하면 
    # self.num_bev_features이라고 바뀐 output feature map dimension 정보를 명시해 주기만 하면 된다.

  def forward(self, data_dict, **kwargs):
    spatial_features = data_dict['spatial_features']
    ups = []
    ret_dict = {}
    x = spatial_features

    ## upblock (layer 5개) -> deblock (layer 1개)
    for i in range(len(self.blocks)): 
      x = self.blocks[i](x) ## 5개의 layer로 이루어짐
      stride = int(spatial_features.shape[2] / x.shape[2])
      ret_dict[f'spatial_features_{stride}x'] = x
      if len(self.deblocks) > 0:
        ups.append(self.deblocks[i](x))
      else:
        ups.append(x)

    if len(ups) > 1:
      x = torch.cat(ups, dim=1) ## [B, 256, 150, 200]의 크기가 항상 deblock의 output이고 이 두개를 concatenate한다.
    
    elif len(ups) == 1:
      x = ups[0]
    
    if len(self.deblocks) > len(self.blocks):
      x = self.deblocks[-1](x) ## 한번더 decoder block이 있다면!
 
    data_dict['spatial_features_2d'] = x ## [B, 512, H, W] 
    # 원래 spatial_features, 즉 8x spconv bbone의 output을 Z축으로 stacking 한 것은 [B, 256, H, W]였는데 2번의 upsampling된 feature map에 concatenate를 하여서 512가 되었다.
    data_dict.update(ret_dict)
    return data_dict
