#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import os
import copy
from functools import partial
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

from ..models.modules.mobileone import MobileOneBlock
from ..models.modules.replknet import ReparamLargeKernelConv

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "pool_size": None,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "fastvit_t": _cfg(crop_pct=0.9),
    "fastvit_s": _cfg(crop_pct=0.9),
    "fastvit_m": _cfg(crop_pct=0.95),
}


# ---------------------------------------------------------------------------
# Helpers for 2D / 3D switching
# ---------------------------------------------------------------------------

def _get_conv(conv_type: str):
    return nn.Conv3d if conv_type == "3d" else nn.Conv2d


def _get_norm(conv_type: str):
    return nn.BatchNorm3d if conv_type == "3d" else nn.BatchNorm2d


def _get_pool(conv_type: str):
    return nn.AdaptiveAvgPool3d if conv_type == "3d" else nn.AdaptiveAvgPool2d


def _scale_shape(dim: int, conv_type: str):
    """Return layer-scale parameter shape for the given conv dimensionality."""
    return (dim, 1, 1, 1) if conv_type == "3d" else (dim, 1, 1)


def _fuse_conv_bn(seq: nn.Sequential) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse a Conv+BN Sequential into (weight, bias). Works for 2D and 3D."""
    conv, bn = seq[0], seq[1]
    std = (bn.running_var + bn.eps).sqrt()
    extra_dims = conv.weight.dim() - 1  # spatial dims + in_ch dim
    t = (bn.weight / std).reshape(-1, *([1] * extra_dims))
    fused_weight = conv.weight * t
    fused_bias = bn.bias - bn.running_mean * bn.weight / std
    return fused_weight, fused_bias


# ---------------------------------------------------------------------------
# Convolutional stem
# ---------------------------------------------------------------------------

def convolutional_stem(
    in_channels: int,
    out_channels: int,
    inference_mode: bool = False,
    conv_type: str = "2d",
    f_maps=None,
) -> nn.Sequential:
    """Build convolutional stem.

    For 2D uses MobileOneBlock (with structural reparameterization).
    For 3D uses plain Conv3d+BN+GELU blocks.
    """
    if conv_type == "3d":
        f_maps = f_maps or [1, 1, 1]
        Conv = nn.Conv3d
        BN = nn.BatchNorm3d
        ch1, ch2, _ = [int(out_channels * f) for f in f_maps]
        return nn.Sequential(
            Conv(in_channels, ch1, 3, stride=2, padding=1, bias=False),
            BN(ch1),
            nn.GELU(),
            Conv(ch1, ch2, 3, stride=2, padding=1,
                 groups=ch1, bias=False),
            BN(ch2),
            nn.GELU(),
            Conv(ch2, out_channels, 1, bias=False),
            BN(out_channels),
            nn.GELU(),
            Conv(out_channels, out_channels, 1, bias=False),
            BN(out_channels),
            nn.GELU(),
        )

    return nn.Sequential(
        MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
    )


# ---------------------------------------------------------------------------
# MHSA – handles both 4-D (B,C,H,W) and 5-D (B,C,D,H,W) tensors
# ---------------------------------------------------------------------------

class MHSA(nn.Module):
    """Multi-headed Self Attention module.

    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        B, C = shape[0], shape[1]
        N = 1
        for s in shape[2:]:
            N *= s

        x_flat = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)
        qkv = (
            self.qkv(x_flat)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        x_out = x_out.transpose(-2, -1).reshape(shape)
        return x_out


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Convolutional patch embedding layer (2D or 3D)."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        inference_mode: bool = False,
        conv_type: str = "2d",
    ) -> None:
        super().__init__()

        if conv_type == "3d":
            from math import gcd
            Conv = nn.Conv3d
            BN = nn.BatchNorm3d
            padding = patch_size // 2
            dw_groups = gcd(in_channels, embed_dim)
            self.proj = nn.Sequential(
                Conv(in_channels, embed_dim, patch_size, stride=stride,
                     padding=padding, groups=dw_groups, bias=False),
                BN(embed_dim),
                nn.GELU(),
                Conv(embed_dim, embed_dim, 1, bias=False),
                BN(embed_dim),
                nn.GELU(),
            )
        else:
            block = list()
            block.append(
                ReparamLargeKernelConv(
                    in_channels=in_channels,
                    out_channels=embed_dim,
                    kernel_size=patch_size,
                    stride=stride,
                    groups=in_channels,
                    small_kernel=3,
                    inference_mode=inference_mode,
                )
            )
            block.append(
                MobileOneBlock(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    inference_mode=inference_mode,
                    use_se=False,
                    num_conv_branches=1,
                )
            )
            self.proj = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# RepMixer
# ---------------------------------------------------------------------------

class RepMixer(nn.Module):
    """Reparameterizable token mixer (2D or 3D).

    For more details, please refer to our paper:
    `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization
    <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
        self,
        dim,
        kernel_size=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode: bool = False,
        conv_type: str = "2d",
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode
        self.conv_type = conv_type

        Conv = _get_conv(conv_type)

        if inference_mode:
            self.reparam_conv = Conv(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            if conv_type == "3d":
                BN = nn.BatchNorm3d
                self.norm = nn.Sequential(
                    Conv(dim, dim, kernel_size, padding=kernel_size // 2,
                         groups=dim, bias=False),
                    BN(dim),
                )
                self.mixer = nn.Sequential(
                    Conv(dim, dim, kernel_size, padding=kernel_size // 2,
                         groups=dim, bias=False),
                    BN(dim),
                )
            else:
                self.norm = MobileOneBlock(
                    dim,
                    dim,
                    kernel_size,
                    padding=kernel_size // 2,
                    groups=dim,
                    use_act=False,
                    use_scale_branch=False,
                    num_conv_branches=0,
                )
                self.mixer = MobileOneBlock(
                    dim,
                    dim,
                    kernel_size,
                    padding=kernel_size // 2,
                    groups=dim,
                    use_act=False,
                )

            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(
                    layer_scale_init_value * torch.ones(_scale_shape(dim, conv_type)),
                    requires_grad=True,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x

    def reparameterize(self) -> None:
        """Reparameterize mixer and norm into a single conv for efficient inference."""
        if self.inference_mode:
            return

        if self.conv_type == "3d":
            self._reparameterize_3d()
        else:
            self._reparameterize_2d()

        for para in self.parameters():
            para.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")

    def _reparameterize_2d(self) -> None:
        self.mixer.reparameterize()
        self.norm.reparameterize()

        if self.use_layer_scale:
            w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = (
                self.mixer.id_tensor
                + self.mixer.reparam_conv.weight
                - self.norm.reparam_conv.weight
            )
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

    def _reparameterize_3d(self) -> None:
        w_mixer, b_mixer = _fuse_conv_bn(self.mixer)
        w_norm, b_norm = _fuse_conv_bn(self.norm)

        # Build 3D identity tensor (depthwise: groups=dim, input_dim=1)
        input_dim = self.dim // self.dim  # 1 since groups == dim
        id_tensor = torch.zeros_like(w_mixer)
        c = self.kernel_size // 2
        for i in range(self.dim):
            id_tensor[i, i % input_dim, c, c, c] = 1

        if self.use_layer_scale:
            ls = self.layer_scale.unsqueeze(-1)  # (dim, 1, 1, 1) → (dim, 1, 1, 1, 1)
            w = id_tensor + ls * (w_mixer - w_norm)
            b = self.layer_scale.view(-1) * (b_mixer - b_norm)
        else:
            w = id_tensor + w_mixer - w_norm
            b = b_mixer - b_norm

        self.reparam_conv = nn.Conv3d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b


# ---------------------------------------------------------------------------
# ConvFFN
# ---------------------------------------------------------------------------

class ConvFFN(nn.Module):
    """Convolutional FFN Module (2D or 3D)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        conv_type: str = "2d",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels

        Conv = _get_conv(conv_type)
        BN = _get_norm(conv_type)
        # Use smaller depthwise kernel for 3D to keep param count reasonable
        dw_kernel = 3 if conv_type == "3d" else 7
        dw_padding = dw_kernel // 2

        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=dw_kernel,
                padding=dw_padding,
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module("bn", BN(num_features=out_channels))
        self.fc1 = Conv(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = Conv(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# RepCPE
# ---------------------------------------------------------------------------

class RepCPE(nn.Module):
    """Implementation of conditional positional encoding (2D or 3D).

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers
    <https://arxiv.org/pdf/2102.10882.pdf>`_
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        spatial_shape: Union[int, Tuple] = (7, 7),
        inference_mode=False,
        conv_type: str = "2d",
    ) -> None:
        super(RepCPE, self).__init__()
        self.conv_type = conv_type

        if conv_type == "3d":
            # Promote spatial_shape to 3D
            if isinstance(spatial_shape, int):
                spatial_shape = (spatial_shape,) * 3
            elif len(spatial_shape) == 2:
                spatial_shape = (spatial_shape[0],) * 3

            self.spatial_shape = spatial_shape
            self.embed_dim = embed_dim
            self.in_channels = in_channels
            self.groups = embed_dim
            padding = spatial_shape[0] // 2

            if inference_mode:
                self.reparam_conv = nn.Conv3d(
                    in_channels, embed_dim, spatial_shape,
                    stride=1, padding=padding, groups=embed_dim, bias=True,
                )
            else:
                self.pe = nn.Conv3d(
                    in_channels, embed_dim, spatial_shape,
                    stride=1, padding=padding, bias=True, groups=embed_dim,
                )
        else:
            if isinstance(spatial_shape, int):
                spatial_shape = tuple([spatial_shape] * 2)
            assert isinstance(spatial_shape, Tuple), (
                f'"spatial_shape" must by a sequence or int, '
                f"get {type(spatial_shape)} instead."
            )
            assert len(spatial_shape) == 2, (
                f'Length of "spatial_shape" should be 2, '
                f"got {len(spatial_shape)} instead."
            )

            self.spatial_shape = spatial_shape
            self.embed_dim = embed_dim
            self.in_channels = in_channels
            self.groups = embed_dim

            if inference_mode:
                self.reparam_conv = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.embed_dim,
                    kernel_size=self.spatial_shape,
                    stride=1,
                    padding=int(self.spatial_shape[0] // 2),
                    groups=self.embed_dim,
                    bias=True,
                )
            else:
                self.pe = nn.Conv2d(
                    in_channels,
                    embed_dim,
                    spatial_shape,
                    1,
                    int(spatial_shape[0] // 2),
                    bias=True,
                    groups=embed_dim,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            x = self.pe(x) + x
            return x

    def reparameterize(self) -> None:
        if hasattr(self, "reparam_conv"):
            return

        if self.conv_type == "3d":
            input_dim = self.in_channels // self.groups
            kernel_value = torch.zeros(
                (self.in_channels, input_dim, *self.spatial_shape),
                dtype=self.pe.weight.dtype,
                device=self.pe.weight.device,
            )
            c = self.spatial_shape[0] // 2
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, c, c, c] = 1
            w_final = kernel_value + self.pe.weight
            b_final = self.pe.bias

            self.reparam_conv = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias=True,
            )
        else:
            input_dim = self.in_channels // self.groups
            kernel_value = torch.zeros(
                (
                    self.in_channels,
                    input_dim,
                    self.spatial_shape[0],
                    self.spatial_shape[1],
                ),
                dtype=self.pe.weight.dtype,
                device=self.pe.weight.device,
            )
            for i in range(self.in_channels):
                kernel_value[
                    i,
                    i % input_dim,
                    self.spatial_shape[0] // 2,
                    self.spatial_shape[1] // 2,
                ] = 1
            w_final = kernel_value + self.pe.weight
            b_final = self.pe.bias

            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias=True,
            )

        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        for para in self.parameters():
            para.detach_()
        self.__delattr__("pe")


# ---------------------------------------------------------------------------
# RepMixerBlock
# ---------------------------------------------------------------------------

class RepMixerBlock(nn.Module):
    """Implementation of Metaformer block with RepMixer as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision
    <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
        conv_type: str = "2d",
    ):
        super().__init__()

        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
            conv_type=conv_type,
        )

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            conv_type=conv_type,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones(_scale_shape(dim, conv_type)),
                requires_grad=True,
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x


# ---------------------------------------------------------------------------
# AttentionBlock
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    """Implementation of metaformer block with MHSA as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision
    <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        conv_type: str = "2d",
    ):
        super().__init__()

        # Override norm_layer for 3D if caller passed the 2D default
        if conv_type == "3d" and norm_layer is nn.BatchNorm2d:
            norm_layer = nn.BatchNorm3d

        self.norm = norm_layer(dim)
        self.token_mixer = MHSA(dim=dim)

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            conv_type=conv_type,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(_scale_shape(dim, conv_type)),
                requires_grad=True,
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(_scale_shape(dim, conv_type)),
                requires_grad=True,
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x


# ---------------------------------------------------------------------------
# Stage builder
# ---------------------------------------------------------------------------

def basic_blocks(
    dim: int,
    block_index: int,
    num_blocks: List[int],
    token_mixer_type: str,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    act_layer: nn.Module = nn.GELU,
    norm_layer: nn.Module = nn.BatchNorm2d,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
    inference_mode=False,
    conv_type: str = "2d",
) -> nn.Sequential:
    """Build FastViT blocks within a stage."""
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = (
            drop_path_rate
            * (block_idx + sum(num_blocks[:block_index]))
            / (sum(num_blocks) - 1)
        )
        if token_mixer_type == "repmixer":
            blocks.append(
                RepMixerBlock(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode,
                    conv_type=conv_type,
                )
            )
        elif token_mixer_type == "attention":
            blocks.append(
                AttentionBlock(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    conv_type=conv_type,
                )
            )
        else:
            raise ValueError(
                "Token mixer type: {} not supported".format(token_mixer_type)
            )
    blocks = nn.Sequential(*blocks)

    return blocks


# ---------------------------------------------------------------------------
# FastViT
# ---------------------------------------------------------------------------

class FastViT(nn.Module):
    """
    This class implements `FastViT architecture <https://arxiv.org/pdf/2303.14189.pdf>`_

    Supports both 2D images (conv_type='2d') and 3D voxels (conv_type='3d').
    For 3D mode, the minimum supported input spatial size is 32×32×32.
    """

    def __init__(
        self,
        layers,
        token_mixers: Tuple[str, ...],
        embed_dims=None,
        mlp_ratios=None,
        downsamples=None,
        repmixer_kernel_size=3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU,
        num_classes=1000,
        pos_embs=None,
        down_patch_size=7,
        down_stride=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        fork_feat=False,
        init_cfg=None,
        pretrained=None,
        cls_ratio=2.0,
        inference_mode=False,
        conv_type: str = "2d",
        in_channels: int = 3,
        f_maps=None,
        **kwargs,
    ) -> None:

        super().__init__()

        if conv_type not in ("2d", "3d"):
            raise ValueError(f"conv_type must be '2d' or '3d', got '{conv_type}'")

        # Auto-promote norm_layer for 3D when the caller left the 2D default
        if conv_type == "3d" and norm_layer is nn.BatchNorm2d:
            norm_layer = nn.BatchNorm3d

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.conv_type = conv_type

        if pos_embs is None:
            pos_embs = [None] * len(layers)

        # Convolutional stem
        self.patch_embed = convolutional_stem(
            in_channels, embed_dims[0], inference_mode, conv_type=conv_type,
            f_maps=f_maps
            
        )

        # Build the main stages of the network architecture
        network = []
        for i in range(len(layers)):
            # Add position embeddings if requested
            if pos_embs[i] is not None:
                network.append(
                    pos_embs[i](
                        embed_dims[i],
                        embed_dims[i],
                        inference_mode=inference_mode,
                        conv_type=conv_type,
                    )
                )
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                token_mixer_type=token_mixers[i],
                kernel_size=repmixer_kernel_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
                conv_type=conv_type,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break

            # Patch merging/downsampling between stages.
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        in_channels=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                        inference_mode=inference_mode,
                        conv_type=conv_type,
                    )
                )

        self.network = nn.ModuleList(network)

        # For segmentation and detection, extract intermediate output
        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get("FORK_LAST3", None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f"norm{i_layer}"
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            Pool = _get_pool(conv_type)
            self.gap = Pool(output_size=1)

            if conv_type == "3d":
                exp_ch = int(embed_dims[-1] * cls_ratio)
                self.conv_exp = nn.Sequential(
                    nn.Conv3d(
                        embed_dims[-1], exp_ch, kernel_size=3, stride=1, padding=1,
                        groups=embed_dims[-1], bias=False,
                    ),
                    nn.BatchNorm3d(exp_ch),
                    nn.GELU(),
                )
            else:
                self.conv_exp = MobileOneBlock(
                    in_channels=embed_dims[-1],
                    out_channels=int(embed_dims[-1] * cls_ratio),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=embed_dims[-1],
                    inference_mode=inference_mode,
                    use_se=True,
                    num_conv_branches=1,
                )

            self.head = (
                nn.Linear(int(embed_dims[-1] * cls_ratio), num_classes)
                if num_classes > 0
                else nn.Identity()
            )

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)

        # load pre-trained model
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m: nn.Module) -> None:
        """Init. for classification"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _scrub_checkpoint(checkpoint, model):
        sterile_dict = {}
        for k1, v1 in checkpoint.items():
            if k1 not in model.state_dict():
                continue
            if v1.shape == model.state_dict()[k1].shape:
                sterile_dict[k1] = v1
        return sterile_dict

    def init_weights(self, pretrained: str = None) -> None:
        """Init. for mmdetection or mmsegmentation by loading
        ImageNet pre-trained weights.
        """
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warning(
                f"No pre-trained weights for "
                f"{self.__class__.__name__}, "
                f"training start from scratch"
            )
            pass
        else:
            assert "checkpoint" in self.init_cfg, (
                f"Only support "
                f"specify `Pretrained` in "
                f"`init_cfg` in "
                f"{self.__class__.__name__} "
            )
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg["checkpoint"]
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location="cpu")
            if "state_dict" in ckpt:
                _state_dict = ckpt["state_dict"]
            elif "model" in ckpt:
                _state_dict = ckpt["model"]
            else:
                _state_dict = ckpt

            sterile_dict = FastViT._scrub_checkpoint(_state_dict, self)
            state_dict = sterile_dict
            self.load_state_dict(state_dict, False)

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        x = self.conv_exp(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        cls_out = self.head(x)
        return cls_out


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------

@register_model
def fastvit_t8(pretrained=False, **kwargs):
    """Instantiate FastViT-T8 model variant."""
    layers = [2, 2, 4, 2]
    embed_dims = [48, 96, 192, 384]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_t"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


@register_model
def fastvit_t12(pretrained=False, **kwargs):
    """Instantiate FastViT-T12 model variant."""
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_t"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


@register_model
def fastvit_s12(pretrained=False, **kwargs):
    """Instantiate FastViT-S12 model variant."""
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_s"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


@register_model
def fastvit_sa12(pretrained=False, **kwargs):
    """Instantiate FastViT-SA12 model variant."""
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_s"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model



@register_model
def fastvit_mysa(pretrained=False,
    layers = [2, 2, 4, 2],
    embed_dims = [64, 128, 256, 512],
    mlp_ratios = [4, 4, 4, 4] ,
    downsamples = [False, False, True, True],
    spaital_shape = (3, 3, 3),
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention"),
                 **kwargs):
    """Instantiate FastViT-SA08 model variant."""

    # spaital_shape = kwargs.get('spatial_shape', (3, 3, 3))
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=spaital_shape)]
    
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_s"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model










@register_model
def fastvit_sa24(pretrained=False, **kwargs):
    """Instantiate FastViT-SA24 model variant."""
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_s"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


@register_model
def fastvit_sa36(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant."""
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        embed_dims=embed_dims,
        token_mixers=token_mixers,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_m"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


@register_model
def fastvit_ma36(pretrained=False, **kwargs):
    """Instantiate FastViT-MA36 model variant."""
    layers = [6, 6, 18, 6]
    embed_dims = [76, 152, 304, 608]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        embed_dims=embed_dims,
        token_mixers=token_mixers,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_m"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model
