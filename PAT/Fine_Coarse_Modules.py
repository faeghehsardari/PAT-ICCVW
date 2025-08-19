from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch.nn as nn



import torch
import numpy
from einops import rearrange
import torch.nn.functional as F

class Local_Relational_Block(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.TC = nn.Conv1d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features) # k=3, stride=1, padding=1
        self.act = act_layer()
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.linear1(x)
        x = x.transpose(1, 2)
        x = self.TC(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x

class Global_Positional_Relational_Block(nn.Module):
    def __init__(self, dim, num_heads=8, max_len = 256, relative_positional_embedig = True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = None or head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.max_len = max_len
        self.Er = nn.Parameter(torch.randn(max_len, head_dim))
        self.relative_positional_embedig = relative_positional_embedig

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if self.relative_positional_embedig:
            start = self.max_len - N
            Er_t = self.Er[start:, :].transpose(0, 1)
            QEr = torch.matmul(q, Er_t)
            Srel = self.skew(QEr)

            attn = (q @ k.transpose(-2, -1))
            attn = (attn + Srel) * self.scale
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x

class Relative_Positional_Transformer_Block(nn.Module):
    """
    Global Local Relational Block
    """

    def __init__(self, dim, num_heads, max_len, mlp_ratio=4., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, relative_positional_embedig = True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.Global__Positional_Relational_Block = Global_Positional_Relational_Block(
            dim,num_heads=num_heads, max_len = max_len, relative_positional_embedig = relative_positional_embedig)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.Local_Relational_Block = Local_Relational_Block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.Global__Positional_Relational_Block(self.norm1(x))
        x = x + self.Local_Relational_Block(self.norm2(x))
        return x

class Temporal_Merging_Block(nn.Module):
    """
    Temporal_Merging_Block
    """

    def __init__(self, kernel_size=3, stride=1, in_chans=1024, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size// 2))
        # self.proj2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

class Fine_Detection_Module(nn.Module):
    def __init__(self, in_feat_dim=1024, embed_dims=[512, 512, 512, 512],
                 num_head=8, mlp_ratio=8, norm_layer=nn.LayerNorm,
                 num_block=3, num_clips = 256, relative_positional_embedig = True):
        super().__init__()

        # fine features
        self.Temporal_Merging_Block_f = Temporal_Merging_Block(kernel_size=3, stride=1, in_chans=in_feat_dim,
                                              embed_dim=embed_dims[0])
        self.fine = nn.ModuleList([Relative_Positional_Transformer_Block(
            dim=embed_dims[0], num_heads=num_head, max_len = num_clips, mlp_ratio=mlp_ratio,norm_layer=norm_layer, relative_positional_embedig=relative_positional_embedig)
            for i in range(num_block)])
        self.norm_f = norm_layer(embed_dims[0])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs):
        fine_feats = self.Temporal_Merging_Block_f(inputs)
        for i, blk in enumerate(self.fine):
            fine_feats = blk(fine_feats)
        fine_feats = self.norm_f(fine_feats)
        fine_feats = fine_feats.permute(0, 2, 1).contiguous()
        return fine_feats

class Coarse_Detection_Module(nn.Module):
    def __init__(self, in_feat_dim=1024, embed_dims=[512, 512, 512, 512],
                 num_head=8, mlp_ratio=8, norm_layer=nn.LayerNorm,
                 num_block=3, num_clips = 256, relative_positional_embedig = True):
        super().__init__()
        # coarse features
        # level 2
        self.Temporal_Merging_Block_c1 = Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.coarse_1 = nn.ModuleList([Relative_Positional_Transformer_Block(
            dim=embed_dims[1], num_heads=num_head, max_len = num_clips // 2, mlp_ratio=mlp_ratio,norm_layer=norm_layer, relative_positional_embedig=relative_positional_embedig)
            for i in range(num_block)])
        self.norm_c1 = norm_layer(embed_dims[1])

        # level 3
        self.Temporal_Merging_Block_c2 = Temporal_Merging_Block(kernel_size=3, stride=4, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[2])
        self.coarse_2 = nn.ModuleList([Relative_Positional_Transformer_Block(
            dim=embed_dims[2], num_heads=num_head, max_len = num_clips // 4, mlp_ratio=mlp_ratio,norm_layer=norm_layer, relative_positional_embedig=relative_positional_embedig)
            for i in range(num_block)])
        self.norm_c2 = norm_layer(embed_dims[2])

        # level 3
        self.Temporal_Merging_Block_c3 = Temporal_Merging_Block(kernel_size=3, stride=8, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[3])
        self.coarse_3 = nn.ModuleList([Relative_Positional_Transformer_Block(
            dim=embed_dims[3], num_heads=num_head, max_len = num_clips // 8, mlp_ratio=mlp_ratio,norm_layer=norm_layer, relative_positional_embedig=relative_positional_embedig)
            for i in range(num_block)])
        self.norm_c3 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, fine_feats):
        coarse_feats = []

        # coarse features (multi-scale, but non-hierarchical)
        coarse_feats_1 = self.Temporal_Merging_Block_c1(fine_feats)
        for i, blk in enumerate(self.coarse_1):
            coarse_feats_1 = blk(coarse_feats_1)
        coarse_feats_1 = self.norm_c1(coarse_feats_1)
        coarse_feats_1 = coarse_feats_1.permute(0, 2, 1).contiguous()
        coarse_feats.append(coarse_feats_1)

        coarse_feats_2 = self.Temporal_Merging_Block_c2(fine_feats)
        for i, blk in enumerate(self.coarse_2):
            coarse_feats_2 = blk(coarse_feats_2)
        coarse_feats_2 = self.norm_c2(coarse_feats_2)
        coarse_feats_2 = coarse_feats_2.permute(0, 2, 1).contiguous()
        coarse_feats.append(coarse_feats_2)

        coarse_feats_3 = self.Temporal_Merging_Block_c3(fine_feats)
        for i, blk in enumerate(self.coarse_3):
            coarse_feats_3 = blk(coarse_feats_3)
        coarse_feats_3 = self.norm_c3(coarse_feats_3)
        coarse_feats_3 = coarse_feats_3.permute(0, 2, 1).contiguous()
        coarse_feats.append(coarse_feats_3)

        return coarse_feats