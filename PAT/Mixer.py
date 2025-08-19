import torch
import torch.nn as nn
import torch.nn.functional as F


class linear_layer(nn.Module):
    #
    def __init__(self, input_dim=2048, embed_dim=512):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):

    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Mixer(nn.Module):
    def __init__(self, inter_channels, embedding_dim):
        super().__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = inter_channels

        self.linear_c3 = nn.Conv1d(c4_in_channels, embedding_dim, kernel_size=3, padding= 1)
        self.linear_c2 = nn.Conv1d(c3_in_channels, embedding_dim, kernel_size=3, padding= 1)
        self.linear_c1 = nn.Conv1d(c2_in_channels, embedding_dim, kernel_size=3, padding= 1)
        self.linear_fine = nn.Conv1d(c1_in_channels, embedding_dim, kernel_size=3, padding= 1)

    def forward(self, fine_feats, coarse_feats):
        coarse_f1, coarse_f2, coarse_f3 = coarse_feats

        coarse_f3 = self.linear_c3(coarse_f3)
        coarse_f3 = resize(coarse_f3, size=fine_feats.size()[2:],mode='linear',align_corners=False)

        coarse_f2 = self.linear_c2(coarse_f2)
        coarse_f2 = resize(coarse_f2, size=fine_feats.size()[2:],mode='linear',align_corners=False)

        coarse_f1 = self.linear_c1(coarse_f1)
        coarse_f1 = resize(coarse_f1, size=fine_feats.size()[2:],mode='linear',align_corners=False)

        fine_feats = self.linear_fine(fine_feats)

        coarse_feats = coarse_f1 + coarse_f2 + coarse_f3

        return  fine_feats, coarse_feats
