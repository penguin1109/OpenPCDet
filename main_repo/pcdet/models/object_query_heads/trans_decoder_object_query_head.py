from .object_query_head_template import ObjectQueryHeadTemplate

class TransDecoderOQHead(ObjectQueryHeadTemplate):
    def __init__(self, model_dict, bev_channels, point_channels, num_class, **kwargs):
        super(TransDecoderOQHead, self).__init__(model_dict=model_dict)