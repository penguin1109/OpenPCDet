import torch
import torch.nn as nn

def bilinear_interpolate_bev(current_bev_features, x, y):
  """ Args
  current_bev_features: [H, W, C]
  current_x_idxs: [N1+N2.., 1] -> 대신 해당 batch가 아니면 masking 처리 되어 있음
  current_y_idxs: [N1+N2.., 1]
  """
  x0 = torch.floor(x).long()
  x1 = x0+1
  x0 = torch.clamp(x0, 0, current_bev_features.shape[1]-1)
  x1 = torch.clamp(x1, 0, current_bev_features.shape[1]-1)

  y0 = torch.floor(y).long()
  y1 = y0 + 1
  y0 = torch.clamp(y0, 0, current_bev_features.shape[0]-1)
  y1 = torch.clamp(y1, 0, current_bev_features.shape[0]-1)


  ## A(x0, y1) B(x1, y1) C(x1, y0) D(x0, y0)
  alpha = y1-y;beta = y-y0;
  p = x-x0;q=x1-x;

  fa = current_bev_features[y1, x0]
  fb = current_bev_features[y1, x1]
  fc = current_bev_features[y0, x1]
  fd = current_bev_features[y0, x0]

  wa = (x - x0.type_as(x)) * (y1.type_as(y) - y) ## 왼쪽 아래 넓이
  wb = (x1.type_as(x) - x) * (y1.type_as(y) - y) ## 오른쪽 아래 넓이
  wc = (x1.type_as(x) - x) * (y - y0.type_as(y)) ## 오른쪽 위 넓이
  wd = (x - x0.type_as(x)) * (y - y0.type_as(y)) ## 왼쪽 위 넓이

  ans = torch.t((torch.t(fd) * wb)) + torch.t((torch.t(fc)) * wa) + torch.t((torch.t(fa)) * wc) + torch.t((torch.t(fb)) * wd)

  return ans



import os, sys
os.chdir('/content/drive/MyDrive/internship/MLV_Lab/2023_Winter/openpcdet/pcdet')
BASE_DIR='/content/drive/MyDrive/internship/MLV_Lab/2023_Winter/openpcdet'
OPS_DIR=os.path.join(BASE_DIR, 'pcdet', 'ops')
BATCH_DIR=os.path.join(OPS_DIR, 'pointnet2_batch')
STACK_DIR=os.path.join(OPS_DIR, 'pointnet2_stack')
sys.path.append(BATCH_DIR);sys.path.append(STACK_DIR);sys.path.append(OPS_DIR);sys.path.append(BASE_DIR)
from ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils ## 여기서 FPS 알고리즘도 찾을 수 있음
from .set_aggregation import MySetAggregation
import torch
import torch.nn as nn

""" Voxel Set Abstraction

"""
class MyVoxelSetAbstraction(nn.Module):
  def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features, num_rawpoint_features, **kwargs):
    super(MyVoxelSetAbstraction, self).__init__()
    self.model_cfg = model_cfg
    self.voxel_size = voxel_size
    self.point_cloud_range = point_cloud_range
    self.init_sa = self.model_cfg['SA_LAYER']
    SA_CFG=self.model_cfg['SA_LAYER']

    self.SA_layers = nn.ModuleList()
    self.SA_layers_names = []
    self.downsample_times_map = {}
    c_in = 0
    for src_name in self.model_cfg.FEATURES_SOURCE:
      if src_name == 'bev' or src_name == 'raw_points':
        continue
      self.downsample_times_map[src_name] = SA_CFG[src_name].DOWNSAMPLE_FACTOR ## 각각의 convolution layer마다 몇배씩 downsampling 한 결과인지에 대한 내용이다.

      if SA_CFG[src_name].get('INPUT_CHANNELS', None) is None:
        if isinstance(SA_CFG[src_name].MLPS[0], list):
          input_channels = SA_CFG[src_name].MLPS[0][0]
        else:
          input_channels = SA_CFG[src_name].MLPS[0]
      else:
        input_channels = SA_CFG[src_name]['INPUT_CHANNELS']
      
      mlps = SA_CFG[src_name].MLPS
      # print(mlps, input_channels)
      for k in range(len(mlps)):
        mlps[k] = [input_channels] + mlps[k]
      # print(mlps)
      num_c_out = sum(x[-1] for x in mlps)
      cur_layer = MySetAggregation(use_xyz=True, config=SA_CFG[src_name], mlps=mlps)

      self.SA_layers.append(cur_layer)
      self.SA_layers_names.append(src_name)

      c_in += num_c_out
    
    if 'bev' in self.model_cfg.FEATURES_SOURCE:
      c_bev = num_bev_features
      c_in += c_bev
    
    if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
      input_ch = num_rawpoint_features-3
      mlps = SA_CFG['raw_points'].MLPS
      for k in range(len(mlps)):
        mlps[k] = [input_ch] + mlps[k] ## 예를 들면 원래 [[16,16], [16,16]]이었다면 [[1,16,16], [1,16,16]]이렇게 바뀌도록 하는 것이다.
      num_c_out = sum(x[-1] for x in mlps)
      self.SA_rawpoints = MySetAggregation(use_xyz=True, config=SA_CFG['raw_points'], mlps=mlps)
      c_in += num_c_out

    self.vsa_point_feature_fusion = nn.Sequential(
        nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
        nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES), nn.ReLU()
    )
    self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
    self.num_point_features_before_fusion = c_in ## RAW, VL1, VL2, VL3, VL4, BEV의 5개의 부분에서 모두 set abstraction을 해서 가져온 feature들의 channel 수를 의미한다.
    # PVRCNN을 기준으로 하면 [16 + 16 + 32 + 64 + 64 + 256] -> [128]로 바뀌게 되는 것이다. (근데 BEV의 경우에는 2D BackBone을 거친 뒤의 512dim feature map인지 256dim feature map인지 확실하지 않다.)
  
  def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
    """ Args
    keypoints: (N1 + N2 + .., 4) -> 여기서 4개의 dimension인 이유는 [몇번째 batch에 속하는지, x, y, z]의 정보가 저장이 되어 있기 때문이다.
    bev_features: (B, C, H, W)
    """
    x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0] ## 어떤 grid에 속하는지에 대한 keypoint의 projection index 위치
    y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1] ## 어떤 gride에 속하는지에 대한 keypoint의 projection index 위치치

    x_idxs = x_idxs / bev_stride
    y_idxs = y_idxs / bev_stride

    point_bev_features_list = []
    for k in range(batch_size): ## 각 batch마다 따로 계산함함
      current_batch_mask = (keypoints[:, 0] == k)

      current_x_idxs = x_idxs[current_batch_mask]
      current_y_idxs = y_idxs[current_batch_mask]
      current_bev_features = bev_features[k].permute(1, 2, 0) ## [H, W, C]
      point_bev_features = bilinear_interpolate_bev(current_bev_features, current_x_idxs, current_y_idxs)
      point_bev_features_list.append(point_bev_features)
    
    point_bev_features = torch.cat(point_bev_features_list, dim=0)
    return point_bev_features

  def get_sampled_keypoints(self, batch_dict):
    batch_size = batch_dict['batch_size']
    ## sampling할 때에 point의 source는 PVRCNN에서의 Voxel Set Abstraction은 오직 raw_point 뿐이다.
    src_points = batch_dict['points'][:,1:4]
    batch_indices = batch_dict['points'][:, 0].long()

    keypoints_list = []
    for batch_idx in range(batch_size):
      batch_mask = (batch_indices == batch_idx) ## 현재 순서의 batch에 속해있는 point임을 제한하는 mask
      sampled_points = src_points[batch_mask].unsqueeze(dim=0) ## [1, N, 3] -> 여기서 3은 xyz좌표, N은 해당 batch의 point의 개수
      if self.model_cfg.SAMPLE_METHOD == 'FPS':
        cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(
            sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
        ).long()

        if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
          times = int(self.model_cfg.NUM_KEYPOINTS / sampled_points.shape[1]) + 1
          non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
          cur_pt_idxs[0] = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS] ## FPS로 샘플링했을 때 고정해 놓은 keypoint의 개수에 비해서 적은 경우에는 똑같은 point index를 반복해서 넣어줌
        
        keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
        keypoints_list.append(keypoints)

      elif self.model_cfg.SAMPLE_METHOD == 'SEGMENTPS': ## Point Net Layer로 segmentation 학습해서 높은 foreground score을 가진 point sampling
        pass
      elif self.model_cfg.SAMPLE_METHOD == 'DENSITYPS': ## Kernel Density Estimation으로 density probabiility가 높은 point sampling
        pass
      else:
        raise NotImplementedError
      
      # keypoints_list.append(keypoints)

    keypoints = torch.cat(keypoints_list, dim=0) # [N1, N2, .., 4]
    if len(keypoints.shape) == 3:
      batch_idx = torch.arange(batch_size, device = keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1, 1)
      keypoints = torch.cat((batch_idx.float(), keypoints.view(-1, 3)), dim=1)

    return keypoints


  def get_voxel_centers(self, coords, downsample_times, voxel_size, point_cloud_range):
    """ Args
    coords: voxel coordinate [N, 3] -> zyx
    """
    assert coords.shape[1] == 3
    centers = coords[:, [2, 1, 0]].float()
    voxel_size = torch.tensor(voxel_size, device = centers.device).float() * downsample_times ## downsampling함에 따라서 voxel의 크기는 커진다고 생각하면 됨 -> 당연히 voxel의 개수가 줄어듦듦
    pc_range = torch.tensor(point_cloud_range[0:3] , device = centers.device).float()
    voxel_centers = (centers + 0.5) * voxel_size + pc_range

    return voxel_centers

  def aggregate_keypoints_from_one_source(self, batch_size, aggregate_func, xyz, xyz_features, xyz_bs_idxs, new_xyz, new_xyz_batch_cnt):
    """ Args
    batch_size: 그냥 별거 아니고 batch 개수
    aggregate_func: 앞서 정의한 ball query로 grouping하고 그안에 있는 voxel feature들을 사용해서 Conv2d - BN - ReLU를 사용함 (이때 random point sampling은 그냥 처음부터 지정된 NSAMPLE 개수만큼 선택)
    xyz: Voxel Center Point, 아니면 Raw Point [M, 3]
    xyz_features: [M, C] -> 없을 수도
    xyz_bs_idxs: [M, 1] -> 각 point가 몇번째 batch에 속하는지
    new_xyz: [N, 3] -> [b1 , b2 ,.. bn]의 순서로 각 batch에 있는 point coordinate가 순차적으로 지정되어 있다.
    new_xyz_batch_cnt: [batch_size] -> 각 batch에 속하는 sampled keypoint의 개수
    """
    xyz_batch_cnt = xyz.new_zeros(batch_size).int()

    for k in range(batch_size):
      xyz_batch_cnt[k] = (xyz_bs_idxs == k).sum()
    
    pooled_points,source_feature = aggregate_func(xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=xyz_features.contiguous())

    return source_feature


  def forward(self, batch_dict):
    ## (1) FPS던 어떤 방법으로라도 keypoint를 sampling 한다.
    keypoints = self.get_sampled_keypoints(batch_dict)
    ## (2) 서로 다른 source로부터 keypoint의 feature을 aggregate하기 위해서 모은다.
    point_features_list = []
    if 'bev' in self.model_cfg.FEATURES_SOURCE:
      bev_feature = batch_dict['spatial_features'] ## [B, 256, H, W]
      point_bev_features = self.interpolate_from_bev_features(
          keypoints, bev_feature, batch_dict['batch_size'], batch_dict['spatial_features_stride']
      )
      point_features_list.append(point_bev_features)
    
    batch_size = batch_dict['batch_size']
    
    new_xyz = keypoints[:, 1:4].contiguous() # 첫번째 dimension의 정보는 몇번째 index의 batch에 속하는지에 대한 정보이다.
    new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()

    for k in range(batch_size):
      new_xyz_batch_cnt[k] = (keypoints[:, 0] == k).sum() # 각 batch에 속하는 keypoint의 개수
    
    if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
      raw_points = batch_dict['points']

      pooled_features= self.aggregate_keypoints_from_one_source(
          batch_size=batch_size, aggregate_func=self.SA_rawpoints, xyz = raw_points[:, 1:4].contiguous(), xyz_features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None,
          xyz_bs_idxs = raw_points[:, 0], new_xyz = new_xyz, new_xyz_batch_cnt = new_xyz_batch_cnt
      )
      point_features_list.append(pooled_features)
    
    for k, src_name in enumerate(self.SA_layers_names):
      ## multi_scale_3d_features는 8x sparse convolution으로 연산이 된 것이기 때문에 -> SparseConvTensor 객체로서 저장이 되어 있다.
      cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
      cur_features = batch_dict['multi_scale_3d_features'][src_name].features.contiguous()
      xyz = self.get_voxel_centers(
          cur_coords[:, 1:4], downsample_times = self.downsample_times_map[src_name],
          voxel_size=self.voxel_size, point_cloud_range = self.point_cloud_range
      )

      pooled_features = self.aggregate_keypoints_from_one_source(
          batch_size=batch_size, aggregate_func = self.SA_layers[k], xyz= xyz.contiguous(), xyz_features = cur_features,
          xyz_bs_idxs = cur_coords[:, 0], new_xyz = new_xyz, new_xyz_batch_cnt = new_xyz_batch_cnt
      )

      point_features_list.append(pooled_features)
    
    ## (3) 모든 source의 feaure들을 concatenate한다.
    point_features = torch.cat(point_features_list, dim=-1)
    batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
    # print(point_features.shape)
    point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

    batch_dict['point_features'] = point_features # [BxN, C]
    batch_dict['point_coords'] = keypoints # [BxN, 4] -> [#batch_idx, x, y, z]

    ## (4) Model CFG 원상복귀 -> 도대체 와 바뀌는지 알수가 없다.
    for key, value in self.model_cfg['SA_LAYER'].items():
      temp = value['MLPS']
      temp[0] = temp[0][1:];temp[1] = temp[1][1:];
      value['MLPS'] = temp
    return batch_dict







    

