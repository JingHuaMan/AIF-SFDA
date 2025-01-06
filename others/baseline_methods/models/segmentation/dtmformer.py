import torch
import torch.nn as nn
import math
from einops import rearrange
from einops.layers.torch import Rearrange

from others.backbones.unet import Unet
from others.baseline_methods.layers.dtmformer_utils import *
from einops import rearrange

__all__ = ['Setr_DTMFormer']


class Setr_DTMFormer(nn.Module):
    def __init__(
            self, in_chans=1, classes=4, img_size=512, embed_dims=[256, 512],
            num_heads=[4, 8], mlp_ratios=[4, 4], qkv_bias=False, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
            depths=[2, 2], as_depth=[1, 1], total_depth=16, sr_ratios=[8, 4], num_stages=4,
            pretrained=None, dmodel=1024, k=5, sample_ratios=0.125, hidden_size_1=128, hidden_size_2=128,
            patch_size=8, return_map=False):
        super().__init__()

        self.total_depth = total_depth
        self.as_depth = as_depth
        self.depths = depths
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channs = in_chans
        self.k = k
        self.classes = classes
        self.patch_num = int(img_size / patch_size)
        self.dmodel = dmodel

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_chans, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.dmodel),
            Rearrange('b s c -> b c s')
        )

        self.recover = LTR()

        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, self.classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([ATMTransformer(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(as_depth[0])])
        self.norm0_as = norm_layer(embed_dims[0])
        self.ctm_as = ATM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1_as = nn.ModuleList([ATMBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(as_depth[1])])
        self.norm1_as = norm_layer(embed_dims[1])

        self.stage0 = nn.ModuleList([Transformer(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for j in range(depths[0])])
        self.norm0 = norm_layer(embed_dims[0])
        self.ctm1 = ATM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([ATMBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_chans, hidden_size_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size_1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size_1, hidden_size_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size_2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size_2, hidden_size_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size_2),
            nn.ReLU(inplace=True),
        )

        self.skip_connection = Unet(in_chans, hidden_size_2, depth=5, output_mode=Unet.ONE_OUTPUT)

        self.out = nn.Conv2d(hidden_size_2 + classes, classes, kernel_size=1)

    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
        x = self.norm0_as(x)
        self.cur += self.as_depth[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        attnScore = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        attnScore = rearrange(attnScore, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm_as(token_dict, attnScore, ctm_stage=1)

        for j, blk in enumerate(self.stage1_as):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm1_as(token_dict['x'])
        self.cur += self.as_depth[1]
        outs.append(token_dict)

        x = self.recover.forward(outs)

        for i in range(int((self.total_depth - sum(self.as_depth)) / sum(self.depths))):
            outs = []
            for blk in self.stage0:
                x = blk(x, self.patch_num, self.patch_num)
            x = self.norm0(x)
            self.cur += self.depths[0]

            B, N, _ = x.shape
            idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
            agg_weight = x.new_ones(B, N, 1)
            token_dict = {'x': x,
                          'token_num': N,
                          'map_size': [self.patch_num, self.patch_num],
                          'init_grid_size': [self.patch_num, self.patch_num],
                          'idx_token': idx_token,
                          'agg_weight': agg_weight}
            outs.append(token_dict.copy())
            token_dict = self.ctm1(token_dict, attnScore, ctm_stage=i + 2)
            for j, blk in enumerate(self.stage1):
                token_dict = blk(token_dict)
            token_dict['x'] = self.norm1(token_dict['x'])
            self.cur += self.depths[1]
            outs.append(token_dict)
            x = self.recover.forward(outs)

        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.dmodel, self.patch_num, self.patch_num)

        # decoder
        x = self.decoder(x)

        x = torch.concat([self.skip_connection(img), x], dim=1)
        x = self.out(x)

        if self.return_map:
            return [x, attnScore]

        return x


class Setr(nn.Module):
    def __init__(
            self, n_classes=4, patch_size=8, img_size=256, in_chans=1, embed_dims=512,
            num_heads=8, mlp_ratios=4, depths=16, sr_ratios=1):
        super().__init__()

        self.depths = depths
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.in_channs = in_chans

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_chans, embed_dims),
        )
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b s c -> b c s'),
        )

        self.patch_num = int(img_size / patch_size)
        self.mlp_dim = [self.embed_dims * self.mlp_ratios]
        self.dropout = 0.1
        self.num_heads = num_heads
        self.dim_head0 = self.embed_dims / self.num_heads
        self.num_patches = self.patches_height * self.patches_width

        self.transformer = nn.ModuleList([ATMBlock(  # TODO这里修改了
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, sr_ratio=sr_ratios)
            for j in range(depths)])

        self.dmodel = embed_dims
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)

        for blk in self.transformer:
            x = blk(x, self.patch_num, self.patch_num)
        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.dmodel, self.patch_num, self.patch_num)

        x = self.decoder(x)
        return x
