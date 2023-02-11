import torch
import torch.nn as nn

class ObjectQueryHeadTemplate(nn.Module):
    def __init__(self, model_dict):
        super(ObjectQueryHeadTemplate, self).__init__()
