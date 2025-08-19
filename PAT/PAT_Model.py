import torch.nn as nn
from .Classification_Module import Classification_Module
from .Mixer import Mixer
from .Fine_Coarse_Modules import Fine_Detection_Module
from .Fine_Coarse_Modules import Coarse_Detection_Module


class PAT(nn.Module):
    """
    MS-TCT for action detection
    """
    def __init__(self, inter_channels, num_block, head, mlp_ratio, in_feat_dim, final_embedding_dim, num_classes, num_clips):
        super(PAT, self).__init__()

        self.dropout=nn.Dropout()

        self.FDM=Fine_Detection_Module(in_feat_dim=in_feat_dim, embed_dims=inter_channels,
                 num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,num_block=num_block, num_clips=num_clips)

        self.CDM = Coarse_Detection_Module(in_feat_dim=in_feat_dim, embed_dims=inter_channels,
                                         num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,
                                         num_block=num_block, num_clips=num_clips)
        self.Mixer = Mixer(inter_channels=inter_channels, embedding_dim=final_embedding_dim)
        self.Classfication_Module=Classification_Module(num_classes=num_classes, embedding_dim=final_embedding_dim)

    def forward(self, inputs):
        inputs = self.dropout(inputs)

        fine_feats = self.FDM(inputs)
        coarse_feats = self.CDM(fine_feats)
        fine_feats, final_coarse_feats = self.Mixer(fine_feats, coarse_feats)
        fine_probs, coarse_probs = self.Classfication_Module(fine_feats, final_coarse_feats)

        return fine_probs, coarse_probs





