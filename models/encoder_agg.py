import math
import time
import re
import torch
import torch.nn as nn
from functools import partial
from easydict import EasyDict as edict
import yaml

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .vmamba.vmamba import VSSM, LayerNorm2d
from .modules import PCASC
from .modules import IGMAVC
from .modules import DiffEnhance, LoraMoEBlock, FRM, LinearAttention, DetailFeatureExtraction, DenseMambaBlock, \
    ChannelEmbed, CMLLFF, ResMoEBlock, SME
from .freqmamba import VSSBlock, SS2D_map
from engine.logger import get_logger

logger = get_logger()


class DWConv(nn.Module):
    """
    Depthwise convolution bloc: input: x with size(B N C); output size (B N C)
    """

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # B N C -> B C N -> B C H W
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)  # B C H W -> B N C

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        """
        MLP Block: 
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
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
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# compatible with openmmlab
class Backbone_VSSM(VSSM):
    def __init__(self, config, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_extra = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)
            layer_name_extra = f'outnorm_extra{i}'
            self.add_module(layer_name_extra, layer_extra)

        del self.classifier

        self.load_pretrained(pretrained)

        # self.DiffEnhances = nn.ModuleList([
        #     DiffEnhance(in_channels=self.dims[i], out_channels=self.dims[i])
        #     for i in range(self.num_layers)
        # ])

        # self.LoraMoEBlocks = nn.ModuleList([
        #     LoraMoEBlock(config=config, final_embed_dim=self.dims[i])
        #     for i in range(self.num_layers)
        # ])

        self.freqmamba = nn.ModuleList([
            VSSBlock(
                hidden_dim=self.dims[i//2],
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=4.0,
                d_state=8 * 2 ** (i//2),)
            for i in range(self.num_layers)
        ])

        self.map = nn.ModuleList([
            SS2D_map(
                d_model=self.dims[i],
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=4.0,
                d_state=8, )
            for i in range(self.num_layers)
        ])

        self.process = nn.ModuleList([
            nn.Linear(self.dims[i], self.dims[i+1])
            for i in range(self.num_layers - 1)
        ])


        # self.LinearAttentions = nn.ModuleList([
        #     LinearAttention(dim=self.dims[i//2], num_heads=8)
        #     for i in range(self.num_layers * 2)
        # ])

        # self.DetailExtras = nn.ModuleList([
        #     DetailFeatureExtraction(dim=self.dims[i//2])
        #     for i in range(self.num_layers * 2)
        # ])

        # self.DenseMambas = nn.ModuleList([
        #     DenseMambaBlock(dims=self.dims[i//2])
        #     for i in range(self.num_layers * 2)
        # ])

        self.ChannelEmbeds = nn.ModuleList([
            ChannelEmbed(in_channels=self.dims[i] * 2, out_channels=self.dims[i])
            for i in range(self.num_layers)
        ])

        self.CMLlFFS = nn.ModuleList([
            CMLLFF(embed_dims=self.dims[i])
            for i in range(self.num_layers)
        ])

        self.ResMoEBlocks = nn.ModuleList([
            nn.Sequential(
                ResMoEBlock(in_ch=self.dims[i // 2], num_experts=3, topk=1, use_shuffle=True),
                SME(in_ch=self.dims[i // 2]),
            )
            for i in range(self.num_layers * 2)
        ])

        # self.FRMs = nn.ModuleList([
        #     FRM(in_dim=self.dims[i//2], out_dim=self.dims[i//2])
        #     for i in range(self.num_layers * 2)
        # ])

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            pretrained_weights = _ckpt[key]

            new_weights = {}
            for name, param in pretrained_weights.items():
                if name.startswith("layers"):
                    new_weights[name] = param
                    extra_name = name.replace("layers", "layers_extra", 1)
                    new_weights[extra_name] = param
                elif name.startswith("patch_embed"):
                    new_weights[name] = param
                    extra_name = name.replace("patch_embed", "patch_embed_extra", 1)
                    new_weights[extra_name] = param
                else:
                    new_weights[name] = param
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(new_weights, strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x_rgb, x_e):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x_rgb = self.patch_embed(x_rgb)
        x_e = self.patch_embed_extra(x_e)
        outs_rgb = []
        outs_e = []
        outs_semantic = []
        outs_vision = []

        for i, (layer, layer_extra) in enumerate(zip(self.layers, self.layers_extra)):
            b, c, h, w = x_rgb.shape
            o_rgb, x_rgb = layer_forward(layer, x_rgb)  # (B, H, W, C)
            o_e, x_e = layer_forward(layer_extra, x_e)


            o_rgb, o_e = self.CMLlFFS[i](o_rgb, o_e)
            x_rgb, x_e = layer.downsample(o_rgb), layer_extra.downsample(o_e)


            # o_rgb = self.LoraMoEBlocks[i](o_rgb, 'visible') + o_rgb
            # o_e = self.LoraMoEBlocks[i](o_e, 'thermal') + o_e
            if i < 2:
                o_rgb = self.freqmamba[2 * i](o_rgb.permute(0, 2, 3 ,1).view(b, -1, c), (h, w)) + o_rgb
                o_e = self.freqmamba[2 * i + 1](o_e.permute(0, 2, 3 ,1).view(b, -1, c), (h, w)) + o_e

            o_1 = o_rgb.permute(0, 2, 3, 1)
            o_2 = o_e.permute(0, 2, 3, 1)
            o_fused = self.ChannelEmbeds[i](torch.cat([o_1, o_2], dim=3).permute(0, 3, 1, 2))

            o_fused = self.ResMoEBlocks[2 * i](o_fused)
            o_fused= self.ResMoEBlocks[2 * i + 1](o_fused)
            # o_e = self.ResMoEBlocks[4 * i + 2](o_e)
            # o_e = self.ResMoEBlocks[4 * i + 3](o_e)
            # o_rgb = self.FRMs[2 * i](o_rgb) #+ o_rgb
            # o_e = self.FRMs[2 * i + 1](o_e) #+ o_rgb

            # Enhance modal_x
            # if i < 2:
            #     o_e = self.DiffEnhances[i](o_rgb, o_e, 'tde')
            #     x_e = layer_extra.downsample(o_e) + x_e

            # Enhance semantic
            # if i > 1:
            #     o_rgb = self.DiffEnhances[i](o_rgb, o_e, 'rse')
            #     x_rgb = layer.downsample(o_rgb) + x_rgb

            # o_rgb_1 = self.LinearAttentions[2 * i](o_rgb.view(b, c, -1).permute(0, 2 ,1), h, w)
            # o_rgb_2 = self.DetailExtras[2 * i](o_rgb)
            # o_e_1 = self.LinearAttentions[2 * i + 1](o_e.view(b, c, -1).permute(0, 2 ,1), h, w)
            # o_e_2 = self.DetailExtras[2 * i + 1](o_e)

            # o_1 = o_rgb_1 + o_e_1
            # o_2 = o_rgb_2 + o_e_2

            # o_1 = self.DenseMambas[2 * i](o_1.permute(0, 2, 3, 1))
            # o_2 = self.DenseMambas[2 * i + 1](o_2.permute(0, 2, 3, 1))

            # o_1 = o_rgb.permute(0, 2, 3, 1)
            # o_2 = o_e.permute(0, 2, 3, 1)
            # o_fused = self.ChannelEmbeds[i](torch.cat([o_1, o_2], dim=3).permute(0, 3, 1, 2))

            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out_rgb = norm_layer(o_rgb)
                norm_layer_extra = getattr(self, f'outnorm_extra{i}')
                out_e = norm_layer_extra(o_e)
                if not self.channel_first:
                    out_rgb = out_rgb.permute(0, 3, 1, 2)
                    out_e = out_e.permute(0, 3, 1, 2)

                outs_vision.append(out_rgb.contiguous())
                outs_vision.append(out_e.contiguous())
                outs_semantic.append(o_fused.contiguous())

        if len(self.out_indices) == 0:
            return x_rgb, x_e

        return outs_vision, outs_semantic


# def load_config(yaml_path):
#     with open(yaml_path, "r") as f:
#         config = yaml.safe_load(f)
#     return edict(config)
# config_path = '../configs/config_mfnet.yaml'
# config = load_config(config_path)
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# x_rgb = torch.randn(2, 3, 480, 640).to(device)
# x_e = torch.randn(2, 3, 480, 640).to(device)
# channel_first = True
# mamba_model = Backbone_VSSM(
#     config=config,
#     pretrained='/home/suguilin/CMXFS/pretrained/classification/vssm1_tiny_0230s_ckpt_epoch_264.pth',
#     depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, 
#             patch_size=4, in_chans=3, num_classes=1000, 
#             ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
#             ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
#             ssm_init="v0", forward_type="v05_noz", 
#             mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
#             patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
#             downsample_version="v3", patchembed_version="v2", 
#             use_checkpoint=False, posembed=False, imgsize=224,).to(device)
# outs_rgb, outs_e = mamba_model(x_rgb, x_e)
# print(outs_rgb[-1].shape)
# print(outs_e[-1].shape)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

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
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """
    Transformer Block: Self-Attention -> Mix FFN -> OverLap Patch Merging
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
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
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # B C H W
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # B H*W/16 C
        x = self.norm(x)

        return x, H, W


class RGBXTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.extra_patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                                    embed_dim=embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                    in_chans=embed_dims[0],
                                                    embed_dim=embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                    in_chans=embed_dims[1],
                                                    embed_dim=embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                    in_chans=embed_dims[2],
                                                    embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        self.extra_block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.extra_norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.extra_block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + 1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.extra_norm2 = norm_layer(embed_dims[1])

        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.extra_block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.extra_norm3 = norm_layer(embed_dims[2])

        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.extra_block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.extra_norm4 = norm_layer(embed_dims[3])

        cur += depths[3]

        self.IGMAVCs = nn.ModuleList([
            IGMAVC(dim=embed_dims[0], reduction=4),
            IGMAVC(dim=embed_dims[1], reduction=4),
            IGMAVC(dim=embed_dims[2], reduction=4),
            IGMAVC(dim=embed_dims[3], reduction=4)])

        self.PCASCs = nn.ModuleList([
            PCASC(dim=embed_dims[0], num_heads=num_heads[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  attn_drop=attn_drop_rate, proj_drop=drop_rate, sr_ratio=sr_ratios[0]),
            PCASC(dim=embed_dims[1], num_heads=num_heads[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  attn_drop=attn_drop_rate, proj_drop=drop_rate, sr_ratio=sr_ratios[1]),
            PCASC(dim=embed_dims[2], num_heads=num_heads[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  attn_drop=attn_drop_rate, proj_drop=drop_rate, sr_ratio=sr_ratios[2]),
            PCASC(dim=embed_dims[3], num_heads=num_heads[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  attn_drop=attn_drop_rate, proj_drop=drop_rate, sr_ratio=sr_ratios[3])])

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

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x_rgb, x_e):
        """
        x_rgb: B x N x H x W
        """
        B = x_rgb.shape[0]
        outs_semantic = []
        outs_vision = []

        # stage 1
        x_rgb, H, W = self.patch_embed1(x_rgb)
        # B H*W/16 C
        x_e, _, _ = self.extra_patch_embed1(x_e)
        for i, blk in enumerate(self.block1):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block1):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm1(x_rgb)
        x_e = self.extra_norm1(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.IGMAVCs[0](x_rgb, x_e)
        x_fused = self.PCASCs[0](x_rgb, x_e)

        outs_vision.append(x_rgb)
        outs_vision.append(x_e)
        outs_semantic.append(x_fused)

        # stage 2
        x_rgb, H, W = self.patch_embed2(x_rgb)
        x_e, _, _ = self.extra_patch_embed2(x_e)
        for i, blk in enumerate(self.block2):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block2):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm2(x_rgb)
        x_e = self.extra_norm2(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.IGMAVCs[1](x_rgb, x_e)
        x_fused = self.PCASCs[1](x_rgb, x_e)

        outs_vision.append(x_rgb)
        outs_vision.append(x_e)
        outs_semantic.append(x_fused)

        # stage 3
        x_rgb, H, W = self.patch_embed3(x_rgb)
        x_e, _, _ = self.extra_patch_embed3(x_e)
        for i, blk in enumerate(self.block3):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block3):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm3(x_rgb)
        x_e = self.extra_norm3(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.IGMAVCs[2](x_rgb, x_e)
        x_fused = self.PCASCs[2](x_rgb, x_e)

        outs_vision.append(x_rgb)
        outs_vision.append(x_e)
        outs_semantic.append(x_fused)

        # stage 4
        x_rgb, H, W = self.patch_embed4(x_rgb)
        x_e, _, _ = self.extra_patch_embed4(x_e)
        for i, blk in enumerate(self.block4):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block4):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm4(x_rgb)
        x_e = self.extra_norm4(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.IGMAVCs[3](x_rgb, x_e)
        x_fused = self.PCASCs[3](x_rgb, x_e)

        outs_vision.append(x_rgb)
        outs_vision.append(x_e)
        outs_semantic.append(x_fused)

        return outs_vision, outs_semantic

    def forward(self, x_rgb, x_e):
        out_vision, out_semantic = self.forward_features(x_rgb, x_e)
        return out_vision, out_semantic


def load_dualpath_model(model, model_file):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
            state_dict[k.replace('patch_embed', 'extra_patch_embed')] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
            state_dict[k.replace('block', 'extra_block')] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v
            state_dict[k.replace('norm', 'extra_norm')] = v

    t_ioend = time.time()

    model.load_state_dict(state_dict, strict=False)
    del state_dict

    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))


class vmamba_tiny(Backbone_VSSM):
    def __init__(self, config=None, channel_first=True, **kwargs):
        super(vmamba_tiny, self).__init__(
            config=config,
            pretrained='/home/suguilin/CMXFS/pretrained/classification/vssm1_tiny_0230s_ckpt_epoch_264.pth',
            depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2,
            patch_size=4, in_chans=3, num_classes=1000,
            ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
            ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
            ssm_init="v0", forward_type="v05_noz",
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
            patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
            downsample_version="v3", patchembed_version="v2",
            use_checkpoint=False, posembed=False, imgsize=224,
        )


# def load_config(yaml_path):
#     with open(yaml_path, "r") as f:
#         config = yaml.safe_load(f)
#     return edict(config)
# config_path = '../configs/config_mfnet.yaml'
# config = load_config(config_path)
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# x_rgb = torch.randn(2, 3, 480, 640).to(device)
# x_e = torch.randn(2, 3, 480, 640).to(device)
# channel_first = True
# mamba_model = vmamba_tiny(config=config, channel_first=channel_first).to(device)
# outs_rgb, outs_e = mamba_model(x_rgb, x_e)
# print(outs_rgb[1].shape)
# print(outs_e[-1].shape)

class mit_b0(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b1(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
