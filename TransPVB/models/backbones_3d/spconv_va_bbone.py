## 8x sparse convolutional network with Voxel Attention
import torch.nn as nn
from functools import partial
!pip install einops
from einops import rearrange
from pcdet.utils.spconv_utils import spconv #, replace_feature
from pcdet.utils.common_utils import get_voxel_centers

def post_act_block(in_ch, out_ch, ksize, indice_key=None, stride=1, padding=0, conv_type='subm', norm_fn=None):
  if conv_type == 'subm':
    conv = spconv.SubMConv3d(in_ch, out_ch, ksize, bias=False, indice_key=indice_key)
  elif conv_type == 'spconv':
    conv = spconv.SparseConv3d(in_ch, out_ch, ksize, bias=False, indice_key=indice_key, padding=padding, stride=stride)
  elif conv_type == 'inverseconv':
    conv = spconv.SparseInverseConv3d(in_ch, out_ch, ksize, indice_key=indice_key,bias=False)
  else:
    raise NotImplementedError 

  m = spconv.SparseSequential(
      conv, norm_fn(out_ch), nn.ReLU()
  )
  return m

class VoxelAttention(nn.Module):
  def __init__(self, model_cfg, voxel_size, point_cloud_range, in_ch, sparse_shape):
    super(VoxelAttention, self).__init__()
    self.model_cfg = model_cfg
    self.voxel_size = voxel_size
    self.point_cloud_range = point_cloud_range
    self.in_ch = in_ch
    self.sparse_shape = sparse_shape
    self.fc = nn.Sequential(
        nn.Linear(self.in_ch + 3, self.in_ch), nn.ReLU(), nn.Sigmoid()
    )
  
  def forward(self, batch_dict, x):
    voxel_centers = get_voxel_centers(
        voxel_coords=batch_dict['voxel_coords'], downsample_times=2, voxel_size=self.voxel_size, point_clount_range=self.point_cloud_range
    ) ## [N, 1, 3]
    if len(voxel_centers.shape) == 2:
      voxel_centers = torch.unsqueeze(voxel_centers, dim=1)
    
    x = x.dense()
    B, C, D, H, W = x.shape ## 원래 sparse tensor이기 때문에 dense tensor로 바꿔 주어야 한다. (Batch Size, Channel dim, Depth, Height, Width)
    
    reshaped = rearrange(x, 'b c d h w -> b (w h d) 1 c',w=W,h=H,d=D,c=C,b=B)
    concat = torch.cat([reshaped, x], dim=-1)
    concat = self.fc(concat)
    attn_weight = rearrange(concat, 'b (w h d) 1 c -> b w h d c', w=W,h=H,d=D, c=C)
    attn_out = torch.mul(x, attn_weight)
    attn_out = rearrange(attn_out, 'b w h d c -> b c d h w', b=B,c=C,d=D,h=H,w=W) ## 다시 sparse tensor로 바꾸기 위해서 shape를 변형해 주어야 한다.

    attn_out_sp_tensor = spconv.SparseConvTensor(
        features=attn_out, ## VFE의 output이다.
        indices=batch_dict['voxel_coords'].int(),
        spatial_shape=self.sparse_shape,
        batch_size=B
    )
    return attn_out_sp_tensor





class VoxelAttentionBackBone8x(nn.Module):
  def __init__(self, model_cfg, grid_size, input_channels, voxel_size, point_cloud_range, **kwargs):
    super(VoxelAttentionBackBone8x, self).__init__()
    self.model_cfg = model_cfg
    norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

    self.sparse_shape = grid_size[::-1] + [1, 0, 0] ## KITTI 데이터셋을 기준으로 하면 ..?

    self.conv_input = spconv.SparseSequential(
        spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
        norm_fn(16), nn.ReLU()
    ) ## [B, 16, D, H, W]
 
    block = post_act_block ## Convolution을 한 다음에 activation을 취하게 된다.
    
    self.conv1 = spconv.SparseSequential(
        block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
    )

    self.conv2 = spconv.SparseSequential(
        # [1600, 1408, 41] -> [800, 704, 21] (kitti dataset기준으로 [Y, X, Z]이다.)
        block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2')
    )

    self.voxel_attention = VoxelAttention(
        model_cfg, voxel_size, point_cloud_range, in_ch=32, sparse_shape=torch.ceil(self.sparse_shape/2)
    )

    self.conv3 = spconv.SparseSequential(
        # [800, 704, 21] -> [400, 352, 11]
        block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
    )

    self.conv4 = spconv.SparseSequential(
        # [400, 352, 11] -> [200, 176, 5]
        block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0,1,1), indice_key='spconv4', conv_type='spconv'),
        block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
    )

    last_pad= 0
    last_pad = self.model_cfg.get('last_pad', last_pad)
    self.conv_last = spconv.SparseSequential(
        spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'), ## [B, 128, 2, 150, 200] = [B, C, D, H, W]
        norm_fn(128), nn.ReLU()
    )
    self.backbone_channels = {
        'x_conv1' : 16, 'x_conv2': 32, 'x_conv3': 64, 'x_conv4': 64
    }
    self.num_point_features = 128

  
  def forward(self, batch_dict):
    voxel_features = batch_dict['voxel_features'] # torch.tensor(batch_dict['voxel_features'])
    voxel_coords = batch_dict['voxel_coords']
    batch_size = batch_dict['batch_size']
    input_sp_tensor = spconv.SparseConvTensor(
        features=voxel_features, ## VFE의 output이다.
        indices=voxel_coords.int(),
        spatial_shape=self.sparse_shape, ## spatial shape는 전체 point cloud range에서 voxel 크기만큼 나누었을 때 얻을 수 있는 x,y,z별 voxel의 개수로 구한 정보이다.
        batch_size=batch_size
    ) 

    x = self.conv_input(input_sp_tensor)
    x_conv1 = self.conv1(x)
    x_conv2 = self.conv2(x_conv1)
    ## Voxel Attention을 첫번째 Downsampling layer이후에 사용한다.
    attn_out_sp_tensor = self.voxel_attention(batch_dict, x=x_conv2)
    x_conv3 = self.conv3(x_conv2)
    x_conv4 = self.conv4(x_conv3)
    ## 얘는 이제 BEV map으로 바꾸어 주기 위해서 3D_to_BEV모듈에 넣어 주어야 한다.
    out = self.conv_last(x_conv4)

    batch_dict['encoded_spconv_stride'] = 0
    batch_dict['encoded_spconv_tensor'] = 0
    batch_dict['multi_scale_3d_features'] = 0
    batch_dict['multi_scale_3d_strides'] = 0

    batch_dict.update({
        'encoded_spconv_tensor' : out, 'encoded_spconv_stride': 8
    })
    batch_dict.update({
        'multi_scale_3d_features' : {
            'x_conv1': x_conv1, 'x_conv2': x_conv2, 'x_conv3': x_conv3, 'x_conv4': x_conv4
        }
    })
    batch_dict.update({
        'multi_scale_3d_strides': {
            'x_conv1': 1, 'x_conv2': 2, 'x_conv3': 4, 'x_conv4' : 8
        }
    })

    return batch_dict
