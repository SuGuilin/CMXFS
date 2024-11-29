import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from einops.layers.torch import Rearrange
from typing import Tuple, List
import math
import sys 
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easydict import EasyDict as edict
import yaml
from utils.mamba_utils import VSSBlock, Permute
from utils.utils import LayerNorm

##########################################################################
## Multi-branch Differential Bidirectional Fusion Network for RGB-T Semantic Segmentation
class DiffEnhance(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(DiffEnhance, self).__init__()
        self.local_attention = nn.Sequential(
            nn.Conv2d(1, 4 * reduction, kernel_size=3, padding=1, stride=1),
            # nn.BatchNorm2d(4 * reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * reduction, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.feature_transform = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, tir, mode="tde"):
        if mode == "tde":
            return self.forward_tde(rgb, tir)
        elif mode == "rse":
            return self.forward_rse(rgb, tir)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward_tde(self, rgb, tir):
        rgb_max_out, _ = torch.max(rgb, dim=1, keepdim=True)
        local_mask = self.local_attention(rgb_max_out)
        global_mask = self.global_attention(tir)

        tir_enhanced = tir * local_mask * global_mask

        fused_input = torch.cat((rgb, tir), dim=1)
        fused_features = self.feature_transform(fused_input)
        fused_features = fused_features * local_mask + tir_enhanced
        return fused_features

    def forward_rse(self, rgb, tir):
        tir_max_out, _ = torch.max(tir, dim=1, keepdim=True)
        local_mask = self.local_attention(tir_max_out)

        rgb_refined = rgb * local_mask
        tir_refined = tir * local_mask + rgb_refined

        rgb_enhanced = self.global_attention(rgb_refined) * rgb_refined
        tir_enhanced = self.global_attention(tir_refined) * tir_refined

        return rgb_enhanced + tir_enhanced


##########################################################################
## Multi-Task Dense Prediction via Mixture of Low-Rank Experts
class ChannelAtt(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            # nn.BatchNorm2d(num_feat // squeeze_factor),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
            # nn.BatchNorm2d(num_feat),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = avg_out + max_out
        out = x * self.sigmoid(attn)
        return out


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=16):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAtt(num_feat, squeeze_factor),
        )

    def forward(self, x):
        x = self.cab(x)
        return x


class LoraBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, rank=6):
        super().__init__()
        self.W = nn.Conv2d(in_channels, rank, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.M = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1)

    def init_weights(self):
        nn.init.kaiming_uniform_(self.W.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W.bias)
        nn.init.kaiming_uniform_(self.M.weight, a=math.sqrt(5))
        nn.init.zeros_(self.M.bias)
    
    def forward(self, x):
        x = self.W(x)
        x = self.M(x)
        return x


class SpatialAtt(nn.Module):
    def __init__(self, dim, dim_out, im_size): # im_size=H*W
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(dim_out)
        self.convsp = nn.Conv2d(dim, dim, kernel_size=1) #nn.Linear(im_size, 1)
        self.ln_sp = nn.LayerNorm(dim)
        self.conv2 = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.conv3 = nn.Conv2d(dim_out, dim_out, kernel_size=1)
    
    def forward(self, x): # output: (n, dim_out*2, 1, 1)
        n, _, h, w = x.shape
        feat = self.conv1(x)
        feat = self.ln(feat.reshape(n, -1, h * w).permute(0, 2, 1)).permute(0, 2, 1).reshape(n, -1, h, w)
        feat = self.act(feat)
        feat = self.conv3(feat)

        feat_sp = torch.mean(x, dim=(2, 3), keepdim=True)
        feat_sp = self.convsp(feat_sp)
        feat_sp = self.ln_sp(feat_sp.reshape(n, -1)).reshape(n, -1, 1, 1)
        # feat_sp = self.convsp(x.reshape(n, -1, h * w)).reshape(n, 1, -1)
        # feat_sp = self.ln_sp(feat_sp).reshape(n, -1, 1, 1)
        feat_sp = self.act(feat_sp)
        feat_sp = self.conv2(feat_sp)
        
        n, c, h, w = feat.shape
        feat = torch.mean(feat.reshape(n, c, h * w), dim=2).reshape(n, c, 1, 1)
        feat = torch.cat([feat, feat_sp], dim=1)

        return feat

class LoraMoEBlock(nn.Module):
    def __init__(self, config, final_embed_dim, im_size=120*160, kernel_size=3):
        super().__init__()
        self.num_lora = len(config.rank_list)
        self.config = config
        self.lora_list_1 = nn.ModuleList()
        rank_list = config.rank_list
        for i in range(self.num_lora):
            self.lora_list_1.append(LoraBlock(final_embed_dim, final_embed_dim, kernel_size=kernel_size, rank=rank_list[i]))
            self.lora_list_1[i].init_weights()
        self.conv1 = nn.ModuleDict()
        self.conv2 = nn.ModuleDict()
        self.conv3 = nn.ModuleDict()
        self.bn = nn.ModuleDict()
        self.bn_all = nn.ModuleDict()
        self.share_conv = nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=3, padding=1)
        self.router_1 = nn.ModuleDict() 
        self.activate = nn.GELU()
        for modal in self.config.modals:
            self.conv1[modal] =  nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=1)   #CAB(final_embed_dim)
            self.conv3[modal] = nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=1)
            self.conv2[modal] = LoraBlock(final_embed_dim, final_embed_dim, kernel_size=kernel_size, rank=config.spe_rank)

            self.bn[modal] = nn.BatchNorm2d(final_embed_dim)
            self.bn_all[modal] = nn.BatchNorm2d(final_embed_dim)

        self.pre_softmax = config.pre_softmax
        self.desert_k = len(config.rank_list) - config.topk
        for modal in self.config.modals:
            self.router_1[modal] = nn.ModuleList()
            self.router_1[modal].append(SpatialAtt(final_embed_dim, final_embed_dim // 4, im_size=im_size))
            self.router_1[modal].append(nn.Conv2d(final_embed_dim // 2, self.num_lora * 2 + 1, kernel_size=1))
        
    def forward(self, x, modal):
        out_ori = self.conv1[modal](x)
        out = out_ori
        n, c, h, w = out.shape
        route_feat = self.router_1[modal][0](out)
        prob_all = self.router_1[modal][1](route_feat).unsqueeze(2)
        prob_lora, prob_mix = prob_all[:, :self.num_lora * 2], prob_all[:, self.num_lora * 2:]
        route_1_raw, stdev_1 = prob_lora.chunk(2, dim=1)  # n, 15, 1, 1, 1
        if self.training:
            noise = torch.randn_like(route_1_raw) * stdev_1
        else:
            noise = 0
        if self.pre_softmax:
            route_1_raw = route_1_raw + noise
            route_1_indice = torch.topk(route_1_raw, self.desert_k, dim=1, largest=False)[1]
            # Set unselected expert gate values ​​to negative infinity
            for j in range(n):
                for i in range(self.desert_k):
                    route_1_raw[j, route_1_indice[j, i].reshape(-1)] = -1e10
            route_1 = torch.softmax(route_1_raw, dim=1)
        else:
            route_1_raw = torch.softmax(route_1_raw + noise, dim=1)
            route_1_indice = torch.topk(route_1_raw, self.desert_k, dim=1, largest=False)[1]
            route_1 = route_1_raw.clone()
            for j in range(n):
                for i in range(self.desert_k):
                    route_1[j, route_1_indice[j, i].reshape(-1)] = 0
        
        lora_out_1 = []
        for i in range(self.num_lora):
            lora_out_1.append(self.lora_list_1[i](out).unsqueeze(1)) # n, 1, c, h, w
        lora_out_1 = torch.cat(lora_out_1, dim=1)
        lora_out_1 = torch.sum(lora_out_1 * route_1, dim=1)
        
        out = self.bn_all[modal](lora_out_1) + self.conv2[modal](out) * prob_mix[:, 0] + self.share_conv(out.detach())
        out = self.bn[modal](out)
        out = self.activate(out)
        out = self.conv3[modal](out)
        return out



##########################################################################
## Linear Attention
class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding."""

    def __init__(self, dim, base=10000):
        super(RoPE, self).__init__()
        self.base = base
        self.dim = dim

    def forward(self, x):
        shape = x.shape[-3:-1]  # Get the last two dimensions as shape
        channel_dims = shape
        feature_dim = self.dim
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (self.base ** (torch.arange(k_max) / k_max))
        angles = torch.cat(
            [
                t.unsqueeze(-1) * theta_ks
                for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing="ij")
            ], dim=-1,).to(x.device)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)

        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class LinearAttention(nn.Module):
    r"""Linear Attention with LePE and RoPE. 

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(dim=dim)

    def forward(self, x, h, w):
        """
        Args:
            x: input features with shape of (B, N, C)
            h: height
            w: width
        """
        b, n, c = x.shape

        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n**-0.5)) @ (v * (n**-0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}"


##########################################################################
## Detail Feature Extraction
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=in_dim, oup=out_dim, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=in_dim, oup=out_dim, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=in_dim, oup=out_dim, expand_ratio=2)
        self.shffleconv = nn.Conv2d(
            in_dim * 2, in_dim * 2, kernel_size=1, stride=1, padding=0, bias=True
        )

    def separateFeature(self, x):
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 : x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, dim, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode(in_dim=dim // 2, out_dim=dim // 2) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
        # self.reduce_conv = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x):
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 : x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        z = torch.cat((z1, z2), dim=1)
        # z = self.reduce_conv(z)
        return z

##########################################################################
## Channel Dimensionality Reduction Learning
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleChannel(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_chans, in_chans, 1)
        self.conv2 = nn.Conv2d(in_chans, out_chans*2, kernel_size=3, padding=1, stride=1, groups=out_chans)
        self.sg = SimpleGate()
        self.sigmoid = nn.Sigmoid()
        self.mlp = nn.Sequential(
                nn.Conv2d(2, 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 1, 1), 
                nn.Sigmoid()
            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        short_x = x
        x = self.conv2(x * self.sigmoid(self.conv1(self.avg_pool(x)) + self.conv1(self.max_pool(x))))
        x = self.sg(x)
        x_mean_out = torch.mean(short_x, dim=1, keepdim=True)
        x_max_out, _ = torch.max(short_x, dim=1, keepdim=True)
        sp = self.mlp(torch.cat([x_mean_out, x_max_out], dim=1))
        # x = x * sp + x
        x = x * (1 + sp)
        return x


##########################################################################
## Mamba Block
class DenseMambaBlock(nn.Module):
    def __init__(
        self,
        depths=[1, 1, 1], 
        dims=96, 
        # =========================
        d_state=16, 
        # =========================
        drop_rate=0., 
        drop_path_rate=0.1, 
        norm_layer=nn.LayerNorm,
        **kwargs,
        ):
        super().__init__()
        self.num_layers = len(depths)
        self.id = id
        if isinstance(dims, int):
            dims = [int(dims) for i_layer in range(self.num_layers)]
        self.dims = dims
        self.embed_dim = dims[0]
        self.d_state = d_state if d_state is not None else math.ceil(dims[0] / 6)
        # [start, end, steps]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.apply(self._init_weights)
        self.vssm, self.conv = self._create_modules(depths, dims, dpr, norm_layer, d_state, drop_rate)

    def _create_modules(self, depths, dims, dpr, norm_layer, d_state, drop_rate):
        vssm = nn.ModuleList(
            self._make_layer(
                dim=dims[i],
                depth=depths[i],
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                d_state=d_state,
                drop_rate=drop_rate,
            ) for i in range(self.num_layers)
        )
        conv = nn.ModuleList(
            nn.Sequential(
                Permute(0, 3, 1, 2),
                (nn.Conv2d(dims[i] * (i + 2), dims[i], 1) 
                if i != self.num_layers - 1 else SimpleChannel(dims[i] * (i + 2), dims[i])),
                nn.ReLU(),
                Permute(0, 2, 3, 1),
            ) for i in range(self.num_layers)
        )
        return vssm, conv

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)    

    @staticmethod
    def _make_layer(dim=32, depth=1, drop_path=[0.1, 0.1], norm_layer=nn.LayerNorm, d_state=16, drop_rate=0.0, **kwargs,):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(
                VSSBlock(hidden_dim=dim, drop_path=drop_path[d], norm_layer=norm_layer, d_state=d_state, drop=drop_rate, **kwargs,)
            )
        return nn.Sequential(*blocks)
    
    def forward(self, modal):
        outs = []

        outs.append(modal)
        for i in range(self.num_layers):
            modal = self.vssm[i](modal)
            modal = self.conv[i](torch.cat([modal, outs[-1]], dim=3))
            outs.append(torch.cat([modal, outs[-1]], dim=3))
        modal = modal + outs[0]
        return modal


##########################################################################
## https://github.com/huaaaliu/RGBX_Semantic_Segmentation/blob/e251d860aebc2f583a6c4919877e6bebe7f1aff3/models/net_utils.py#L136

class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True),
                        norm_layer(out_channels) 
                        )
        self.norm = norm_layer(out_channels)
        
    def forward(self, x):
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out



class CMLLFF(nn.Module):
    def __init__(self, embed_dims, squeeze_factor=16, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(embed_dims*2, embed_dims*2 // squeeze_factor, 1, bias=False),
            # nn.BatchNorm2d(embed_dims*2 // squeeze_factor),
            # nn.SiLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims*2 // squeeze_factor, embed_dims, 1, bias=False), 
            # nn.BatchNorm2d(embed_dims),
            nn.Sigmoid(),
            )
            for _ in range(2)
        ])

        self.mlp = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(2, 4 * reduction, kernel_size=3, padding=1, stride=1),
            # nn.BatchNorm2d(4 * reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * reduction, 1, 1), 
            nn.Sigmoid()
            )
            for _ in range(2)
        ])

        self.gate = nn.Sequential(
                    nn.Linear(embed_dims * 2, embed_dims * 2 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dims * 2 // reduction, embed_dims),
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, C, H, W = x1.shape

        x1_flat = x1.flatten(2).transpose(1, 2)  ##B H*W C
        x2_flat = x2.flatten(2).transpose(1, 2)  ##B H*W C
        gated_weight = self.gate(torch.cat((x1_flat, x2_flat), dim=2))  ##B H*W C
        gated_weight = gated_weight.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()  ## B C H W

        gap_x1 = self.avg_pool(x1)
        gmp_x1 = self.max_pool(x1)
        ap_x1 = torch.mean(x1, dim=1, keepdim=True)
        mp_x1, _ = torch.max(x1, dim=1, keepdim=True) 
        gp_x1 = torch.cat([gap_x1, gmp_x1], dim=1)
        p_x1 = torch.cat([ap_x1, mp_x1], dim=1)
        gp_x1_ca = self.fc[0](gp_x1)
        p_x1_sp = self.mlp[0](p_x1)

        gap_x2 = self.avg_pool(x2)
        gmp_x2 = self.max_pool(x2)
        ap_x2 = torch.mean(x2, dim=1, keepdim=True)
        mp_x2, _ = torch.max(x2, dim=1, keepdim=True) 
        gp_x2 = torch.cat([gap_x2, gmp_x2], dim=1)
        p_x2 = torch.cat([ap_x2, mp_x2], dim=1)
        gp_x2_ca = self.fc[1](gp_x2)
        p_x2_sp = self.mlp[1](p_x2)

        out_x1 = (x2 * gp_x1_ca + x2 * p_x1_sp) * gated_weight + x1
        out_x2 = (x1 * gp_x2_ca + x1 * p_x2_sp) * (1 - gated_weight) + x2
        
        return out_x1, out_x2 


class FRM(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, down_kernel=5, down_stride=4):
        super().__init__()
        self.project_in = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.lconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.hconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.norm1 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.down = nn.Conv2d(in_dim, in_dim, kernel_size=down_kernel, stride=down_stride, padding=down_kernel//2, groups=in_dim)
        self.proj = nn.Conv2d(in_dim*2, out_dim, kernel_size=1, stride=1, padding=0)
        self.MM = nn.ModuleList([
            nn.Conv2d(in_dim, in_dim, (1, 7), padding=(0, 3), groups=in_dim),
            nn.Conv2d(in_dim, in_dim, (1, 11), padding=(0, 5), groups=in_dim),
            nn.Conv2d(in_dim, in_dim, (1, 21), padding=(0, 10), groups=in_dim),
            nn.Conv2d(in_dim, in_dim, (7, 1), padding=(3, 0), groups=in_dim),
            nn.Conv2d(in_dim, in_dim, (11, 1), padding=(5, 0), groups=in_dim),
            nn.Conv2d(in_dim, in_dim, (21, 1), padding=(10, 0), groups=in_dim),
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        x_scale = 0
        for conv in self.MM:
            x_scale = x_scale + conv(x)
        x = x + x_scale

        dx = self.down(x)  # (H/4, W/4)
        udx = F.interpolate(dx, size=(H, W), mode='bilinear', align_corners=False)
        lx = self.norm1(self.lconv(self.act(x * udx)))
        hx = self.norm2(self.hconv(self.act(x - udx)))
        out = self.act(self.proj(torch.cat([lx, hx], dim=1))) 
        return out


##########################################################################
## Low-rank mixture of experts Block
def channel_shuffle(x, groups=2):
    batch_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(batch_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, w, h)
    return x

class MoEBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_experts: int,
        topk: int,
        use_shuffle: bool = True,
        lr_space: str = "linear",
        recursive: int = 2,
    ):
        super().__init__()
        self.use_shuffle = use_shuffle
        self.recursive = recursive

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, 2 * in_ch, kernel_size=1, padding=0),
        )

        self.agg_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=4, padding=2, groups=in_ch), nn.GELU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
        )

        self.conv_2 = nn.Sequential(
            StripedConv2d(in_ch, kernel_size=3, depthwise=True), nn.GELU()
        )

        if lr_space == "linear":
            grow_func = lambda i: i + 2
        elif lr_space == "exp":
            grow_func = lambda i: 2 ** (i + 1)
        elif lr_space == "double":
            grow_func = lambda i: 2 * i + 2
        else:
            raise NotImplementedError(f"lr_space {lr_space} not implemented")

        self.moe_layer = MoELayer(
            experts=[
                Expert(in_ch=in_ch, low_dim=grow_func(i)) for i in range(num_experts)
            ],  # add here multiple of 2 as low_dim
            gate=Router(in_ch=in_ch, num_experts=num_experts),
            num_expert=topk,
        )

        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def calibrate(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x

        for _ in range(self.recursive):
            x = self.agg_conv(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return res + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)

        if self.use_shuffle:
            x = channel_shuffle(x, groups=2)
        x, k = torch.chunk(x, chunks=2, dim=1)

        x = self.conv_2(x)
        k = self.calibrate(k)

        x = self.moe_layer(x, k)
        x = self.proj(x)
        return x


class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_expert: int = 1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert = num_expert

    def forward(self, inputs: torch.Tensor, k: torch.Tensor):
        out = self.gate(inputs)
        weights = F.softmax(out, dim=1, dtype=torch.float).to(inputs.dtype)
        topk_weights, topk_experts = torch.topk(weights, self.num_expert)
        # normalize the weights of the selected experts
        # topk_weights = F.softmax(topk_weights, dim=1, dtype=torch.float).to(inputs.dtype)
        out = inputs.clone()

        if self.training:
            exp_weights = torch.zeros_like(weights)
            exp_weights.scatter_(1, topk_experts, weights.gather(1, topk_experts))
            for i, expert in enumerate(self.experts):
                out += expert(inputs, k) * exp_weights[:, i : i + 1, None, None]
        else:
            selected_experts = [self.experts[i] for i in topk_experts.squeeze(dim=0)]
            for i, expert in enumerate(selected_experts):
                out += expert(inputs, k) * topk_weights[:, i : i + 1, None, None]

        return out


class Expert(nn.Module):
    def __init__(
        self,
        in_ch: int,
        low_dim: int,
    ):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv2d(low_dim, in_ch, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(k) * x  # here no more sigmoid
        x = self.conv_3(x)
        return x


class Router(nn.Module):
    def __init__(self, in_ch: int, num_experts: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange("b c 1 1 -> b c"),
            nn.Linear(in_ch, num_experts, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class StripedConv2d(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int, depthwise: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=(1, self.kernel_size),
                padding=(0, self.padding),
                groups=in_ch if depthwise else 1,
            ),
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=(self.kernel_size, 1),
                padding=(self.padding, 0),
                groups=in_ch if depthwise else 1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GatedFFN(nn.Module):
    def __init__(self, in_ch, mlp_ratio, kernel_size, act_layer,):
        super().__init__()
        mlp_ch = in_ch * mlp_ratio
        
        self.fn_1 = nn.Sequential(
            nn.Conv2d(in_ch, mlp_ch, kernel_size=1, padding=0),
            act_layer,
        )
        self.fn_2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
            act_layer,
        )
        
        self.gate = nn.Conv2d(mlp_ch // 2, mlp_ch // 2, 
                              kernel_size=kernel_size, padding=kernel_size // 2, groups=mlp_ch // 2)

    def feat_decompose(self, x):
        s = x - self.gate(x)
        x = x + self.sigma * s
        return x
    
    def forward(self, x: torch.Tensor):
        x = self.fn_1(x)
        x, gate = torch.chunk(x, 2, dim=1)
        
        gate = self.gate(gate)
        x = x * gate
        
        x = self.fn_2(x)
        return x


class SME(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int = 11):
        super().__init__()
        
        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.block = StripedConvFormer(in_ch=in_ch, kernel_size=kernel_size)
    
        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
        self.ffn = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm_1(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x


class StripedConvFormer(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
        self.to_qv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, padding=0),
            nn.GELU(),
        )

        self.attn = StripedConv2d(in_ch, kernel_size=kernel_size, depthwise=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, v = self.to_qv(x).chunk(2, dim=1)
        q = self.attn(q)
        x = self.proj(q * v)
        return x


class ResMoEBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_experts: int,
        topk: int,
        lr_space: int = 2,
        recursive: int = 2,
        use_shuffle: bool = False,
    ):
        super().__init__()
        lr_space_mapping = {1: "linear", 2: "exp", 3: "double"}
        self.norm = LayerNorm(in_ch, data_format="channels_first")
        self.block = MoEBlock(
            in_ch=in_ch,
            num_experts=num_experts,
            topk=topk,
            use_shuffle=use_shuffle,
            recursive=recursive,
            lr_space=lr_space_mapping.get(lr_space, "linear"),
        )
        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
        self.ffn = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# def load_config(yaml_path):
#         with open(yaml_path, "r") as f:
#             config = yaml.safe_load(f)
#         return edict(config)
# config_path = '../configs/config_mfnet.yaml'
# config = load_config(config_path)
# rgb = torch.randn(2, 96, 120, 160).to(device)
# tir = torch.randn(2, 96, 120, 160).to(device)
# model = ResMoEBlock(in_ch=96, num_experts=4, topk=2, use_shuffle=True).to(device)
# y1 = model(rgb)
# y2 = model(tir)
# print(y1.shape)
# print(y2.shape)

#########################################################################################  
###########  Interactive Gated Mix Attention module for Visual Completion ###############
######################################################################################### 

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 2, self.dim * 2 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 2 // reduction, self.dim * 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg_v = self.avg_pool(x).view(B, self.dim * 2) #B  2C
        max_v = self.max_pool(x).view(B, self.dim * 2)
        
        avg_se = self.mlp(avg_v).view(B, self.dim * 2, 1)
        max_se = self.mlp(max_v).view(B, self.dim * 2, 1)
        
        Stat_out = self.sigmoid(avg_se+max_se).view(B, self.dim * 2, 1)
        channel_weights = Stat_out.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super(SpatialAttention, self).__init__()
        self.mlp = nn.Sequential(
                    nn.Conv2d(4, 4*reduction, kernel_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(4*reduction, 2, kernel_size), 
                    nn.Sigmoid())
    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x1_mean_out = torch.mean(x1, dim=1, keepdim=True) #B  1  H  W
        x1_max_out, _ = torch.max(x1, dim=1, keepdim=True)  #B  1  H  W
        x2_mean_out = torch.mean(x2, dim=1, keepdim=True) #B  1  H  W
        x2_max_out, _ = torch.max(x2, dim=1, keepdim=True)  #B  1  H  W                
        x_cat = torch.cat((x1_mean_out, x1_max_out,x2_mean_out,x2_max_out), dim=1) # B 4 H W
        spatial_weights = self.mlp(x_cat).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return spatial_weights


class MixAttention(nn.Module):
    def __init__(self, dim, reduction=1):
        super(MixAttention, self).__init__()
        self.dim = dim
        self.ca_gate = ChannelAttention(self.dim) 
        self.sa_gate = SpatialAttention(reduction=4)

    def forward(self, x1,x2):
        ca_out = self.ca_gate(x1,x2) # 2 B C 1 1
        sa_out = self.sa_gate(x1,x2)  # 2 B 1 H W
        mixatt_out = ca_out.mul(sa_out)  # 2 B C H W
        return mixatt_out


#### Interactive Gated Mix Attention module for Visual Completion ####       
class IGMAVC(nn.Module):
    def __init__(self, dim, reduction=4):
        super(IGMAVC, self).__init__()                         
        #self.gate = nn.Linear(2*dim, 2*dim)
        self.MA = MixAttention(dim)
        self.gate = nn.Sequential(
                    nn.Linear(dim * 2, dim * 2 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim * 2 // reduction, dim),
                    nn.Sigmoid())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = x2.shape
        assert B1 == B2 and C1 == C2 and H1 == H2 and W1 == W2, "x1 and x2 should have the same dimensions"
        x1_flat = x1.flatten(2).transpose(1, 2)  ##B HXW C
        x2_flat = x2.flatten(2).transpose(1, 2)  ##B HXW C
        
        gated_weight = self.gate(torch.cat((x1_flat, x2_flat), dim=2))  ##B HXW C
        gated_weight = gated_weight.reshape(B1, H1, W1, C1).permute(0, 3, 1, 2).contiguous() # B C H W
        
        mix_map = self.MA(x1,x2) #2 B C H W 
        
        Gated_attention_x1 = gated_weight*mix_map[0]
        Gated_attention_x2 = (1-gated_weight)*mix_map[1]
                
        out_x1 = x1 + Gated_attention_x2 * x2  # B C H W
        out_x2 = x2 + Gated_attention_x1 * x1  # B C H W
               
        return out_x1, out_x2  


#########################################################################################  
###########  Progressive Cycle Attention module for Semantic Completion ################# 
######################################################################################### 

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(SelfAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
            
        self.apply(self._init_weights)
              
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # B N C -> B N num_head C//num_head -> B C//num_head N num_heads
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W) 
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) 
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        #kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_atten = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_atten+ x) 
        x_out = self.proj_drop(x_out)

        return x_out


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.sr_ratio = sr_ratio
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear embedding
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)                    
        self.attn_drop = nn.Dropout(attn_drop)        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)


    def forward(self, x1, x2, H, W):
        B1, N1, C1 = x1.shape
        B2, N2, C2 = x2.shape
        assert B1 == B2 and C1 == C2 and N1 == N2, "x1 and x2 should have the same dimensions" 

        # B N C -> B N num_head C//num_head -> B C//num_head N num_heads
        q1 = self.q1(x1).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3) 

        if self.sr_ratio > 1:
            x2_ = x2.permute(0, 2, 1).reshape(B2, C2, H, W) 
            x2_ = self.sr(x2_).reshape(B2, C2, -1).permute(0, 2, 1) 
            x2_ = self.norm(x2_)
            kv2 = self.kv2(x2_).reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4) 
        else:
            kv2 = self.kv2(x2).reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4) 
    
        #kv2 = self.kv2(x2).reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4) 
        k2, v2 = kv2[0], kv2[1]

        attn = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_atten = (attn @ v2).transpose(1, 2).reshape(B2, N2, C2)
        x_out = self.proj(x_atten+x1)
        x_out = self.proj_drop(x_out)

        return x_out

###########  Progressive Cycle Attention module for Smantic Completion ##############################     
class PCASC(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(PCASC, self).__init__()                         
        self.SA_x1 = SelfAttention(dim, num_heads=num_heads,qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.CA_x1toX2 = CrossAttention(dim,num_heads=num_heads,qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.SA_x2 = SelfAttention(dim,num_heads=num_heads,qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.CA_x2toX1 = CrossAttention(dim,num_heads=num_heads,qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.proj = nn.Linear(dim, dim)
        #self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)
              
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = x2.shape
        assert B1 == B2 and C1 == C2 and H1 == H2 and W1 == W2, "x1 and x2 should have the same dimensions"
        x1_flat = x1.flatten(2).transpose(1, 2)  ##B HXW C
        x2_flat = x2.flatten(2).transpose(1, 2)  ##B HXW C
        x1_self_enhance =  self.SA_x1(x1_flat,H1, W1)
        x2_cross_enhance = self.CA_x1toX2(x2_flat,x1_self_enhance,H1, W1)
        x2_self_enhance = self.SA_x2(x2_cross_enhance,H1, W1)
        x1_cross_enhance = self.CA_x2toX1(x1_self_enhance,x2_self_enhance,H1, W1)  ##B HXW C
        Fuse = self.proj(x1_cross_enhance)   ##B HXW C
        #Fuse = self.proj_drop(Fuse)

        Fuse_out = Fuse.permute(0, 2, 1).reshape(B1, C1, H1, W1).contiguous()
          
        return Fuse_out  