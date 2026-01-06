# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Type


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB_a(nn.Module):
    """
    输入 NCHW，输出 NCHW，通道 out_planes
    """
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super().__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False),
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False),
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False),
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False),
        )

        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


class MLPBlock(nn.Module):
    """
    名字不变；输入输出 shape 不变：
      - (B, N, C) -> (B, N, C)  (TwoWayTransformer 用)
      - (B, H, W, C) -> (B, H, W, C)  (ViT Block 用)

    用 BasicRFB_a 替代原来的 Linear-Act-Linear。
    """
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,                 # 保留签名，不破坏外部调用
        act: Type[nn.Module] = nn.GELU,  # 保留签名，不破坏外部调用
    ) -> None:
        super().__init__()
        self.adapter = BasicRFB_a(
            in_planes=embedding_dim,
            out_planes=embedding_dim,
            stride=1,
            scale=1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # (B, N, C) -> (B, C, N, 1) -> RFB -> (B, N, C)
            b, n, c = x.shape
            y = x.permute(0, 2, 1).contiguous().unsqueeze(-1)   # B,C,N,1
            y = self.adapter(y)                                 # B,C,N,1
            y = y.squeeze(-1).permute(0, 2, 1).contiguous()     # B,N,C
            return y

        if x.dim() == 4:
            # (B, H, W, C) -> (B, C, H, W) -> RFB -> (B, H, W, C)
            y = x.permute(0, 3, 1, 2).contiguous()              # B,C,H,W
            y = self.adapter(y)                                 # B,C,H,W
            y = y.permute(0, 2, 3, 1).contiguous()              # B,H,W,C
            return y

        raise ValueError(f"MLPBlock expects 3D or 4D input, got shape {tuple(x.shape)}")


# 必须保留：image_encoder.py 还在 import LayerNorm2d
# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/.../models/convnext.py#L119
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
