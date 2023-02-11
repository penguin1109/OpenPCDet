import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):
  def __init__(self, model_cfg, **kwargs):
    super(TransformerDecoderLayer, self).__init__()
    """
    1. Query, Key, Value 모두 BEV Map
    2. Query: BEV Map  [B, HW, C]
       Key, Value: Keypoint Feature Vector [B, N, C]
    """
    

class TransformerDecoder(nn.Module):
  def __init__(self, model_cfg, **kwargs):
    super(TransformerDecoder, self).__init__()
    self.model_cfg = model_cfg
    self.make_position_encoding()
    
  def make_position_encoding(self):
    """ Position Encoding Vector을 만드는 부분
    - BEV Map에 더해주게 된다. [B, HW]의 크기를 가지게 되고, [B, HW, C]의 크기의 BEV Feature Map에 Pixel-Wise Addition을 한다.
    - 그냥 Cos-Sin하는 것처럼, 2D이기 때문에 똑같이 하고, self.register_buffer('pe', pe)로 적용을 하면 된다.
      -> 별도의 깊이 정보나 density 정보를 추가하는 것도 고려 하자자
    """
    pass
