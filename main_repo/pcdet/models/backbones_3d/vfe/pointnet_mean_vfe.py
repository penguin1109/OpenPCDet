from .vfe_template import VFETemplate
import torch
import torch.nn as nn

class ShallowPNet(nn.Module):
    def __init__()
class PNetMeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        # self.model_cfg = model_cfg -> 이건 실제 VFETemplate에서 정의 해 둠
        self.num_point_features = num_point_features
    
    def get_output_feature_dim(self):
        return self.num_point_features
    
    def forward(self, batch_dict, **kwargs):
        """ Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
        Returns:
            vfe_features: (num_voxels, C)
        """
        pass