# coding=utf-8
# Copyright 2025 SparkAudio & The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch SparkTTS model."""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm # Needed for modules
import torchaudio # Needed for mel transformer in BiCodec
import numpy as np # Needed for BiCodecTokenizer logic

from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any
from collections import namedtuple # For Perceiver
from functools import wraps, partial # For Perceiver/FSQ
from contextlib import nullcontext # For FSQ

from huggingface_hub import snapshot_download
from safetensors.torch import load_file # For BiCodec loading

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast # LLM output type
from transformers.generation import GenerationMixin
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor # Needed for from_pretrained
from transformers.utils import logging
from transformers import AutoTokenizer # Needed for token parser test
from einops import rearrange, repeat, pack, unpack # Needed for modules
from einops.layers.torch import Rearrange # Needed for modules
from packaging import version # Needed for Perceiver

from torch import Tensor, int32, einsum
from torch.amp import autocast
from einops import rearrange, reduce, pack, unpack
from numpy.lib.stride_tricks import sliding_window_view
import soxr
import soundfile

# Import custom config
from .configuration_spark_tts import SparkTTSConfig, SparkTTSBiCodecConfig

logger = logging.get_logger(__name__)

# =============================================================================
# >> START: PASTE CODE FROM sparktts/modules/* HERE <<
# =============================================================================
# IMPORTANT: All classes defined in sparktts/modules/* (layers, samper, vocos,
# fsq, residual_fsq, ecapa_tdnn, pooling_layers, perceiver_encoder,
# speaker_encoder, feat_encoder, feat_decoder, wave_generator,
# factorized_vector_quantize) need to be pasted or defined *within* this file
# so they can be found when `trust_remote_code=True` is used.

# Example placeholder comment:
# --- Paste sparktts/modules/blocks/layers.py content here ---

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


# --- Paste sparktts/modules/blocks/samper.py content here ---
class SamplingBlock(nn.Module):
    """Sampling block for upsampling or downsampling"""

    def __init__(
        self,
        dim: int,
        groups: int = 1,
        upsample_scale: int = 1,
        downsample_scale: int = 1,
    ) -> None:
        """
        Args:
            dim: input dimension
            groups: number of groups
            upsample_scale: upsampling scale
            downsample_scale: downsampling scale
        """
        super(SamplingBlock, self).__init__()

        self.upsample_scale = upsample_scale
        self.downsample_scale = downsample_scale

        if self.upsample_scale > 1:
            self.de_conv_upsampler = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(
                    dim,
                    dim,
                    kernel_size=upsample_scale * 2,
                    stride=upsample_scale,
                    padding=upsample_scale // 2 + upsample_scale % 2,
                    output_padding=upsample_scale % 2,
                    groups=groups,
                ),
            )

        if self.downsample_scale > 1:
            self.conv_downsampler = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv1d(
                    dim,
                    dim,
                    kernel_size=2 * downsample_scale,
                    stride=downsample_scale,
                    padding=downsample_scale // 2 + downsample_scale % 2,
                    groups=groups,
                ),
            )

    @staticmethod
    def repeat_upsampler(x, upsample_scale):
        return x.repeat_interleave(upsample_scale, dim=2)

    @staticmethod
    def skip_downsampler(x, downsample_scale):
        return F.avg_pool1d(x, kernel_size=downsample_scale, stride=downsample_scale)

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.upsample_scale > 1:
            repeat_res = self.repeat_upsampler(x, self.upsample_scale)
            deconv_res = self.de_conv_upsampler(x)
            upmerge_res = repeat_res + deconv_res
        else:
            upmerge_res = x
            repeat_res = x

        if self.downsample_scale > 1:
            conv_res = self.conv_downsampler(upmerge_res)
            skip2_res = self.skip_downsampler(upmerge_res, self.downsample_scale)
            skip1_res = self.skip_downsampler(repeat_res, self.downsample_scale)
        else:
            conv_res = upmerge_res
            skip2_res = upmerge_res
            skip1_res = repeat_res

        final_res = conv_res + skip1_res + skip2_res

        return final_res

# --- Paste sparktts/modules/blocks/vocos.py content here ---
class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.adanorm = condition_dim is not None
        if condition_dim:
            self.norm = AdaLayerNorm(condition_dim, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        condition_dim (int): Dimension of the condition.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, condition_dim: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Linear(condition_dim, embedding_dim)
        self.shift = nn.Linear(condition_dim, embedding_dim)
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding)
        shift = self.shift(cond_embedding)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)
        return x


class ResBlock1(nn.Module):
    """
    ResBlock adapted from HiFi-GAN V1 (https://github.com/jik876/hifi-gan) with dilated 1D convolutions,
    but without upsampling layers.

    Args:
        dim (int): Number of input channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        dilation (tuple[int], optional): Dilation factors for the dilated convolutions.
            Defaults to (1, 3, 5).
        lrelu_slope (float, optional): Negative slope of the LeakyReLU activation function.
            Defaults to 0.1.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=self.get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=self.get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=self.get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
            ]
        )

        self.gamma = nn.ParameterList(
            [
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, gamma in zip(self.convs1, self.convs2, self.gamma):
            xt = torch.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, negative_slope=self.lrelu_slope)
            xt = c2(xt)
            if gamma is not None:
                xt = gamma * xt
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

    @staticmethod
    def get_padding(kernel_size: int, dilation: int = 1) -> int:
        return int((kernel_size * dilation - dilation) / 2)


class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = condition_dim is not None
        if condition_dim:
            self.norm = AdaLayerNorm(condition_dim, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    condition_dim=condition_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        x = self.embed(x)
        if self.adanorm:
            assert condition is not None
            x = self.norm(x.transpose(1, 2), condition)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, condition)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        num_blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """

    def __init__(
        self,
        input_channels,
        dim,
        num_blocks,
        layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = weight_norm(
            nn.Conv1d(input_channels, dim, kernel_size=3, padding=1)
        )
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        self.resnet = nn.Sequential(
            *[
                ResBlock1(dim=dim, layer_scale_init_value=layer_scale_init_value)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x


# --- Paste sparktts/modules/fsq/finite_scalar_quantization.py content here ---
def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# tensor helpers


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


# main class


class FSQ(nn.Module):
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
        num_codebooks=1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices=True,
        force_quantization_f32=True,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )

        self.has_projections = has_projections

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer(
                "implicit_codebook", implicit_codebook, persistent=False
            )

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps: float = 1e-3):
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings"""
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """Inverse of `codes_to_indices`."""
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        if need_move_channel_last:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32
        quantization_context = (
            partial(autocast, "cuda", enabled=False) if force_f32 else nullcontext
        )

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # returning indices could be optional

            indices = None

            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, "b n c d -> b n (c d)")

            codes = codes.type(orig_dtype)

        # project out

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if need_move_channel_last:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")

            indices = maybe(unpack_one)(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, "... 1 -> ...")

        # return quantized output and indices

        return out, indices


# --- Paste sparktts/modules/fsq/residual_fsq.py content here ---
import random
import torch.distributed as dist
from einx import get_at

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def get_maybe_sync_seed(device, max_size=10_000):
    rand_int = torch.randint(0, max_size, (), device=device)

    if is_distributed():
        dist.all_reduce(rand_int)

    return rand_int.item()


class ResidualFSQ(nn.Module):
    """Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf"""

    def __init__(
        self,
        *,
        levels: List[int],
        num_quantizers,
        dim=None,
        is_channel_first=False,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        quantize_dropout_multiple_of=1,
        **kwargs,
    ):
        super().__init__()
        codebook_dim = len(levels)
        dim = default(dim, codebook_dim)

        requires_projection = codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self.has_projections = requires_projection

        self.is_channel_first = is_channel_first
        self.num_quantizers = num_quantizers

        self.levels = levels
        self.layers = nn.ModuleList([])

        levels_tensor = torch.Tensor(levels)

        scales = []

        for ind in range(num_quantizers):
            scales.append((levels_tensor - 1) ** -ind)

            fsq = FSQ(levels=levels, dim=codebook_dim, **kwargs)

            self.layers.append(fsq)

        assert all([not fsq.has_projections for fsq in self.layers])

        self.codebook_size = self.layers[0].codebook_size

        self.register_buffer("scales", torch.stack(scales), persistent=False)

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

    @property
    def codebooks(self):
        codebooks = [layer.implicit_codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks

    def get_codes_from_indices(self, indices):

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices, ps = pack([indices], "b * q")

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert (
                self.quantize_dropout > 0.0
            ), "quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations"
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

        # take care of quantizer dropout

        mask = indices == -1
        indices = indices.masked_fill(
            mask, 0
        )  # have it fetch a dummy code to be masked out later

        all_codes = get_at("q [c] d, b n q -> q b n d", self.codebooks, indices)

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(rearrange(mask, "b n q -> q b n 1"), 0.0)

        # scale the codes

        scales = rearrange(self.scales, "q d -> q 1 1 d")
        all_codes = all_codes * scales

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        (all_codes,) = unpack(all_codes, ps, "q b * d")

        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, "q ... -> ...", "sum")
        return self.project_out(codes_summed)

    def forward(self, x, return_all_codes=False, rand_quantize_dropout_fixed_seed=None):
        num_quant, quant_dropout_multiple_of, device = (
            self.num_quantizers,
            self.quantize_dropout_multiple_of,
            x.device,
        )

        # handle channel first

        if self.is_channel_first:
            x = rearrange(x, "b d ... -> b ... d")
            x, ps = pack([x], "b * d")

        # maybe project in

        x = self.project_in(x)

        quantized_out = 0.0
        residual = x

        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices

        if should_quantize_dropout:

            # check if seed is manually passed in

            if not exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)

            rand = random.Random(rand_quantize_dropout_fixed_seed)

            rand_quantize_dropout_index = rand.randrange(
                self.quantize_dropout_cutoff_index, num_quant
            )

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = (
                    round_up_multiple(
                        rand_quantize_dropout_index + 1, quant_dropout_multiple_of
                    )
                    - 1
                )

            null_indices = torch.full(
                x.shape[:2], -1.0, device=device, dtype=torch.long
            )

        # go through the layers

        with autocast("cuda", enabled=False):
            for quantizer_index, (layer, scale) in enumerate(
                zip(self.layers, self.scales)
            ):

                if (
                    should_quantize_dropout
                    and quantizer_index > rand_quantize_dropout_index
                ):
                    all_indices.append(null_indices)
                    continue

                quantized, indices = layer(residual / scale)

                quantized = quantized * scale

                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                all_indices.append(indices)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # stack all indices

        all_indices = torch.stack(all_indices, dim=-1)

        # channel first out

        if self.is_channel_first:
            (quantized_out,) = unpack(quantized_out, ps, "b * d")
            (all_indices,) = unpack(all_indices, ps, "b * d")

            quantized_out = rearrange(quantized_out, "b ... d -> b d ...")
            all_indices = rearrange(all_indices, "b ... d -> b d ...")

        # return

        ret = (quantized_out, all_indices)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers

        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)


# grouped residual fsq


class GroupedResidualFSQ(nn.Module):
    def __init__(self, *, dim, groups=1, accept_image_fmap=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.groups = groups
        assert (dim % groups) == 0
        dim_per_group = dim // groups

        self.accept_image_fmap = accept_image_fmap

        self.rvqs = nn.ModuleList([])

        for _ in range(groups):
            self.rvqs.append(ResidualFSQ(dim=dim_per_group, **kwargs))

        self.codebook_size = self.rvqs[0].codebook_size

    @property
    def codebooks(self):
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    def get_codes_from_indices(self, indices):
        codes = tuple(
            rvq.get_codes_from_indices(chunk_indices)
            for rvq, chunk_indices in zip(self.rvqs, indices)
        )
        return torch.stack(codes)

    def get_output_from_indices(self, indices):
        outputs = tuple(
            rvq.get_output_from_indices(chunk_indices)
            for rvq, chunk_indices in zip(self.rvqs, indices)
        )
        return torch.cat(outputs, dim=self.split_dim)

    def forward(self, x, return_all_codes=False):
        shape, split_dim, device = x.shape, self.split_dim, x.device
        assert shape[split_dim] == self.dim

        # split the feature dimension into groups

        x = x.chunk(self.groups, dim=split_dim)

        forward_kwargs = dict(
            return_all_codes=return_all_codes,
            rand_quantize_dropout_fixed_seed=(
                get_maybe_sync_seed(device) if self.training else None
            ),
        )

        # invoke residual vq on each group

        out = tuple(rvq(chunk, **forward_kwargs) for rvq, chunk in zip(self.rvqs, x))
        out = tuple(zip(*out))

        # otherwise, get all the zipped outputs and combine them

        quantized, all_indices, *maybe_all_codes = out

        quantized = torch.cat(quantized, dim=split_dim)
        all_indices = torch.stack(all_indices)

        ret = (quantized, all_indices, *maybe_all_codes)
        return ret


# --- Paste sparktts/modules/speaker/pooling_layers.py content here ---

class TAP(nn.Module):
    """
    Temporal average pooling, only first-order mean is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TAP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        pooling_mean = x.mean(dim=-1)
        # To be compatable with 2D input
        pooling_mean = pooling_mean.flatten(start_dim=1)
        return pooling_mean

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TSDP(nn.Module):
    """
    Temporal standard deviation pooling, only second-order std is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSDP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        pooling_std = pooling_std.flatten(start_dim=1)
        return pooling_std

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TSTP(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSTP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim


class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """

    def __init__(self,
                 in_dim,
                 bottleneck_dim=128,
                 global_context_att=False,
                 **kwargs):
        super(ASTP, self).__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim,
                                 kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(
            self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class MHASTP(torch.nn.Module):
    """ Multi head attentive statistics pooling
    Reference:
        Self Multi-Head Attention for Speaker Recognition
        https://arxiv.org/pdf/1906.09890.pdf
    """

    def __init__(self,
                 in_dim,
                 layer_num=2,
                 head_num=2,
                 d_s=1,
                 bottleneck_dim=64,
                 **kwargs):
        super(MHASTP, self).__init__()
        assert (in_dim % head_num
                ) == 0  # make sure that head num can be divided by input_dim
        self.in_dim = in_dim
        self.head_num = head_num
        d_model = int(in_dim / head_num)
        channel_dims = [bottleneck_dim for i in range(layer_num + 1)]
        if d_s > 1:
            d_s = d_model
        else:
            d_s = 1
        self.d_s = d_s
        channel_dims[0], channel_dims[-1] = d_model, d_s
        heads_att_trans = []
        for i in range(self.head_num):
            att_trans = nn.Sequential()
            for i in range(layer_num - 1):
                att_trans.add_module(
                    'att_' + str(i),
                    nn.Conv1d(channel_dims[i], channel_dims[i + 1], 1, 1))
                att_trans.add_module('tanh' + str(i), nn.Tanh())
            att_trans.add_module(
                'att_' + str(layer_num - 1),
                nn.Conv1d(channel_dims[layer_num - 1], channel_dims[layer_num],
                          1, 1))
            heads_att_trans.append(att_trans)
        self.heads_att_trans = nn.ModuleList(heads_att_trans)

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:  # B x F x T
            input = input.reshape(input.shape[0],
                                  input.shape[1] * input.shape[2],
                                  input.shape[3])
        assert len(input.shape) == 3
        bs, f_dim, t_dim = input.shape
        chunks = torch.chunk(input, self.head_num, 1)
        # split
        chunks_out = []
        # for i in range(self.head_num):
        #     att_score = self.heads_att_trans[i](chunks[i])
        for i, layer in enumerate(self.heads_att_trans):
            att_score = layer(chunks[i])
            alpha = F.softmax(att_score, dim=-1)
            mean = torch.sum(alpha * chunks[i], dim=2)
            var = torch.sum(alpha * chunks[i]**2, dim=2) - mean**2
            std = torch.sqrt(var.clamp(min=1e-7))
            chunks_out.append(torch.cat((mean, std), dim=1))
        out = torch.cat(chunks_out, dim=1)
        return out

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class MQMHASTP(torch.nn.Module):
    """ An attentive pooling
    Reference:
        multi query multi head attentive statistics pooling
        https://arxiv.org/pdf/2110.05042.pdf
    Args:
        in_dim: the feature dimension of input
        layer_num: the number of layer in the pooling layer
        query_num: the number of querys
        head_num: the number of heads
        bottleneck_dim: the bottleneck dimension

    SA (H = 1, Q = 1, n = 2, d_s = 1) ref:
        https://www.danielpovey.com/files/2018_interspeech_xvector_attention.pdf
    MHA (H > 1, Q = 1, n = 1, d_s = 1) ref:
        https://arxiv.org/pdf/1906.09890.pdf
    AS (H = 1, Q > 1, n = 2, d_s = 1) ref:
        https://arxiv.org/pdf/1803.10963.pdf
    VSA (H = 1, Q > 1, n = 2, d_s = d_h) ref:
        http://www.interspeech2020.org/uploadfile/pdf/Mon-2-10-5.pdf
    """

    def __init__(self,
                 in_dim,
                 layer_num=2,
                 query_num=2,
                 head_num=8,
                 d_s=2,
                 bottleneck_dim=64,
                 **kwargs):
        super(MQMHASTP, self).__init__()
        self.n_query = nn.ModuleList([
            MHASTP(in_dim,
                   layer_num=layer_num,
                   head_num=head_num,
                   d_s=d_s,
                   bottleneck_dim=bottleneck_dim) for i in range(query_num)
        ])
        self.query_num = query_num
        self.in_dim = in_dim

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:  # B x F x T
            input = input.reshape(input.shape[0],
                                  input.shape[1] * input.shape[2],
                                  input.shape[3])
        assert len(input.shape) == 3
        res = []
        for i, layer in enumerate(self.n_query):
            res.append(layer(input))
        out = torch.cat(res, dim=-1)
        return out

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2 * self.query_num
        return self.out_dim



# --- Paste sparktts/modules/speaker/ecapa_tdnn.py content here ---

class Res2Conv1dReluBn(nn.Module):
    """
    in_channels == out_channels == channels
    """

    def __init__(
        self,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        scale=4,
    ):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias,
                )
            )
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # Order: conv -> relu -> bn
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)

        return out


""" Conv1d + BatchNorm1d + ReLU
"""


class Conv1dReluBn(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


""" The SE connection of 1D case.
"""


class SE_Connect(nn.Module):

    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out


""" SE-Res2Block of the ECAPA-TDNN architecture.
"""


class SE_Res2Block(nn.Module):

    def __init__(self, channels, kernel_size, stride, padding, dilation, scale):
        super().__init__()
        self.se_res2block = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            Res2Conv1dReluBn(
                channels, kernel_size, stride, padding, dilation, scale=scale
            ),
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            SE_Connect(channels),
        )

    def forward(self, x):
        return x + self.se_res2block(x)


class ECAPA_TDNN(nn.Module):

    def __init__(
        self,
        channels=512,
        feat_dim=80,
        embed_dim=192,
        pooling_func="ASTP",
        global_context_att=False,
        emb_bn=False,
    ):
        super().__init__()

        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8
        )
        self.layer3 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8
        )
        self.layer4 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8
        )

        cat_channels = channels * 3
        out_channels = 512 * 3
        self.conv = nn.Conv1d(cat_channels, out_channels, kernel_size=1)
        self.pool = globals()[pooling_func](
            in_dim=out_channels, global_context_att=global_context_att
        )
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)
        self.emb_bn = emb_bn
        if emb_bn:  # better in SSL for SV
            self.bn2 = nn.BatchNorm1d(embed_dim)
        else:
            self.bn2 = nn.Identity()

    def forward(self, x, return_latent=False):
        x = x.permute(0, 2, 1)  # (B,T,F) -> (B,F,T)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        latent = F.relu(self.conv(out))
        out = self.bn(self.pool(latent))
        out = self.linear(out)
        if self.emb_bn:
            out = self.bn2(out)

        if return_latent:
            return out, latent
        return out


def ECAPA_TDNN_c1024(feat_dim, embed_dim, pooling_func="ASTP", emb_bn=False):
    return ECAPA_TDNN(
        channels=1024,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        emb_bn=emb_bn,
    )


def ECAPA_TDNN_GLOB_c1024(feat_dim, embed_dim, pooling_func="ASTP", emb_bn=False):
    return ECAPA_TDNN(
        channels=1024,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        emb_bn=emb_bn,
    )


def ECAPA_TDNN_c512(feat_dim, embed_dim, pooling_func="ASTP", emb_bn=False):
    return ECAPA_TDNN(
        channels=512,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        emb_bn=emb_bn,
    )


def ECAPA_TDNN_GLOB_c512(feat_dim, embed_dim, pooling_func="ASTP", emb_bn=False):
    return ECAPA_TDNN(
        channels=512,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        emb_bn=emb_bn,
    )


# --- Paste sparktts/modules/speaker/perceiver_encoder.py content here ---

def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

# main class


class Attend(nn.Module):
    def __init__(self, dropout=0.0, causal=False, use_flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash = use_flash
        assert not (
            use_flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu
        self.config = namedtuple(
            "EfficientAttentionConfig",
            ["enable_flash", "enable_math", "enable_mem_efficient"],
        )
        self.cpu_config = self.config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once(
                "A100 GPU detected, using flash attention if input tensor is on cuda"
            )
            self.cuda_config = self.config(True, False, False)
        else:
            print_once(
                "Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda"
            )
            self.cuda_config = self.config(False, True, True)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = rearrange(k, "b ... -> b 1 ...").expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, "b ... -> b 1 ...").expand_as(q)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )

        return out

    def forward(self, q, k, v, mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v, mask=mask)

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        # similarity

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out


def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))


class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True, dim_cond=None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond=None):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim=-1) * self.scale * gamma

        if not self.cond:
            return out

        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma, beta = map(lambda t: rearrange(t, "b d -> b 1 d"), (gamma, beta))
        return out * gamma + beta


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (kernel_size,) = self.kernel_size
        (dilation,) = self.dilation
        (stride,) = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value=0.0)
        return super().forward(causal_padded_x)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, causal_conv=False):
    dim_inner = int(dim * mult * 2 / 3)

    conv = None
    if causal_conv:
        conv = nn.Sequential(
            Rearrange("b n d -> b d n"),
            CausalConv1d(dim_inner, dim_inner, 3),
            Rearrange("b d n -> b n d"),
        )

    return Sequential(
        nn.Linear(dim, dim_inner * 2), GEGLU(), conv, nn.Linear(dim_inner, dim)
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        dropout=0.0,
        use_flash=False,
        cross_attn_include_queries=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(causal=causal, dropout=dropout, use_flash=use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x, context=None, mask=None):
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        out = self.attend(q, k, v, mask=mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=2,
        dim_context=None,
        num_latents=32,
        dim_head=64,
        heads=8,
        ff_mult=4,
        use_flash_attn=False,
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.proj_context = (
            nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()
        )

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            use_flash=use_flash_attn,
                            cross_attn_include_queries=True,
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        batch = x.shape[0]

        x = self.proj_context(x)

        latents = repeat(self.latents, "n d -> b n d", b=batch)

        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


# --- Paste sparktts/modules/speaker/speaker_encoder.py content here ---

class SpeakerEncoder(nn.Module):
    """

    Args:
        input_dim (int): acoustic feature dimension
        out_dim (int): output dimension of x-vector and d-vector
        latent_dim (int): latent dimension before quantization
        token_num (int): sequence length of speaker tokens
        fsq_levels (List[int]): number of levels for each quantizer
        fsq_num_quantizers (int): number of quantizers

    Return:
        speaker_embs: (B, T2, out_dim)
    """

    def __init__(
        self,
        input_dim: int = 100,
        out_dim: int = 512,
        latent_dim: int = 128,
        token_num: int = 32,
        fsq_levels: List[int] = [4, 4, 4, 4, 4, 4],
        fsq_num_quantizers: int = 1,
    ):
        super(SpeakerEncoder, self).__init__()

        self.speaker_encoder = ECAPA_TDNN_GLOB_c512(
            feat_dim=input_dim, embed_dim=out_dim
        )
        self.perceiver_sampler = PerceiverResampler(
            dim=latent_dim, dim_context=512 * 3, num_latents=token_num
        )
        self.quantizer = ResidualFSQ(
            levels=fsq_levels,
            num_quantizers=fsq_num_quantizers,
            dim=latent_dim,
            is_channel_first=True,
            quantize_dropout=False,
        )

        self.project = nn.Linear(latent_dim * token_num, out_dim)

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        zq = self.quantizer.get_codes_from_indices(indices.transpose(1, 2))
        return zq.transpose(1, 2)

    def get_indices(self, mels: torch.Tensor) -> torch.Tensor:
        mels = mels.transpose(1, 2)
        x = self.perceiver_sampler(mels).transpose(1, 2)
        zq, indices = self.quantizer(x)
        return indices

    def forward(self, mels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mels: (B, D_mel, T1)

        Return:
            x_vector: (B, out_dim)
            d_vector: (B, out_dim)
        """
        # mels = mels.transpose(1,2)

        x_vector, features = self.speaker_encoder(mels, True)
        x = self.perceiver_sampler(features.transpose(1, 2)).transpose(1, 2)
        zq, indices = self.quantizer(x)  # zq: (B, latent_dim, T2, latent_dim)
        x = zq.reshape(zq.shape[0], -1)
        d_vector = self.project(x)

        return x_vector, d_vector
    
    def tokenize(self, mels: torch.Tensor) -> torch.Tensor:
        """tokenize the input mel spectrogram"""
        _, features = self.speaker_encoder(mels, True)
        x = self.perceiver_sampler(features.transpose(1, 2)).transpose(1, 2)
        zq, indices = self.quantizer(x)
        return indices
    
    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        """detokenize the input indices to d-vector"""
        zq = self.quantizer.get_output_from_indices(indices.transpose(1, 2)).transpose(1, 2)
        x = zq.reshape(zq.shape[0], -1)
        d_vector = self.project(x)
        return d_vector


# --- Paste sparktts/modules/encoder_decoder/feat_encoder.py content here ---

class Encoder(nn.Module):
    """Encoder module with convnext and downsampling blocks"""

    def __init__(
        self,
        input_channels: int,
        vocos_dim: int,
        vocos_intermediate_dim: int,
        vocos_num_layers: int,
        out_channels: int,
        sample_ratios: List[int] = [1, 1],
    ):
        super().__init__()
        """
        Encoder module with VocosBackbone and sampling blocks.

        Args:
            sample_ratios (List[int]): sample ratios
                example: [2, 2] means downsample by 2x and then upsample by 2x
        """
        self.encoder = VocosBackbone(
            input_channels=input_channels,
            dim=vocos_dim,
            intermediate_dim=vocos_intermediate_dim,
            num_layers=vocos_num_layers,
            condition_dim=None,
        )

        modules = [
            nn.Sequential(
                SamplingBlock(
                    dim=vocos_dim,
                    groups=vocos_dim,
                    downsample_scale=ratio,
                ),
                VocosBackbone(
                    input_channels=vocos_dim,
                    dim=vocos_dim,
                    intermediate_dim=vocos_intermediate_dim,
                    num_layers=2,
                    condition_dim=None,
                ),
            )
            for ratio in sample_ratios
        ]

        self.downsample = nn.Sequential(*modules)

        self.project = nn.Linear(vocos_dim, out_channels)

    def forward(self, x: torch.Tensor, *args):
        """
        Args:
            x (torch.Tensor): (batch_size, input_channels, length)

        Returns:
            x (torch.Tensor): (batch_size, encode_channels, length)
        """
        x = self.encoder(x)
        x = self.downsample(x)
        x = self.project(x)
        return x.transpose(1, 2)



# --- Paste sparktts/modules/encoder_decoder/feat_decoder.py content here ---

class Decoder(nn.Module):
    """Decoder module with convnext and upsampling blocks

    Args:
        sample_ratios (List[int]): sample ratios
            example: [2, 2] means downsample by 2x and then upsample by 2x
    """

    def __init__(
        self,
        input_channels: int,
        vocos_dim: int,
        vocos_intermediate_dim: int,
        vocos_num_layers: int,
        out_channels: int,
        condition_dim: int = None,
        sample_ratios: List[int] = [1, 1],
        use_tanh_at_final: bool = False,
    ):
        super().__init__()

        self.linear_pre = nn.Linear(input_channels, vocos_dim)
        modules = [
            nn.Sequential(
                SamplingBlock(
                    dim=vocos_dim,
                    groups=vocos_dim,
                    upsample_scale=ratio,
                ),
                VocosBackbone(
                    input_channels=vocos_dim,
                    dim=vocos_dim,
                    intermediate_dim=vocos_intermediate_dim,
                    num_layers=2,
                    condition_dim=None,
                ),
            )
            for ratio in sample_ratios
        ]

        self.downsample = nn.Sequential(*modules)

        self.vocos_backbone = VocosBackbone(
            input_channels=vocos_dim,
            dim=vocos_dim,
            intermediate_dim=vocos_intermediate_dim,
            num_layers=vocos_num_layers,
            condition_dim=condition_dim,
        )
        self.linear = nn.Linear(vocos_dim, out_channels)
        self.use_tanh_at_final = use_tanh_at_final

    def forward(self, x: torch.Tensor, c: torch.Tensor = None):
        """encoder forward.

        Args:
            x (torch.Tensor): (batch_size, input_channels, length)

        Returns:
            x (torch.Tensor): (batch_size, encode_channels, length)
        """
        x = self.linear_pre(x.transpose(1, 2))
        x = self.downsample(x).transpose(1, 2)
        x = self.vocos_backbone(x, condition=c)
        x = self.linear(x).transpose(1, 2)
        if self.use_tanh_at_final:
            x = torch.tanh(x)

        return x


# --- Paste sparktts/modules/encoder_decoder/wave_generator.py content here ---

class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        kernel_size: int = 2,
        stride: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class WaveGenerator(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        kernel_sizes,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, (kernel_size, stride) in enumerate(zip(kernel_sizes, rates)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, kernel_size, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)


# --- Paste sparktts/modules/vq/factorized_vector_quantize.py content here ---

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


class FactorizedVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        commitment: float,
        codebook_loss_weight: float = 1.0,
        decay: float = 0.99,
        threshold_ema_dead_code: float = 2,
        momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.codebook_loss_weight = codebook_loss_weight
        self.decay = decay
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.momentum = momentum

        if input_dim != self.codebook_dim:
            self.in_project = WNConv1d(input_dim, self.codebook_dim, kernel_size=1)
            self.out_project = WNConv1d(self.codebook_dim, input_dim, kernel_size=1)

        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))

    def forward(self, z: torch.Tensor) -> Dict[str, Any]:
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """
        # transpose since we use linear

        # Factorized codes project input into low-dimensional space if self.input_dim != self.codebook_dim
        z_e = self.in_project(z)
        z_q, indices, dists = self.decode_latents(z_e)

        # statistic the usage of codes
        embed_onehot = F.one_hot(indices, self.codebook_size).type(z_e.dtype)
        avg_probs = torch.mean(embed_onehot.reshape(-1, self.codebook_size), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        active_num = (embed_onehot.sum(0).sum(0) > 0).sum()
        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            ema_inplace(self.cluster_size, embed_onehot.sum(0).sum(0), self.decay)
            active_num = sum(self.cluster_size > self.threshold_ema_dead_code)

        if self.training:
            commit_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
                * self.commitment
            )

            codebook_loss = (
                F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
                * self.codebook_loss_weight
            )

        else:
            commit_loss = torch.zeros(0, device=z.device)
            codebook_loss = torch.zeros(0, device=z.device)

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_project(z_q)

        vq_loss = (commit_loss + codebook_loss).mean()

        return {
            "z_q": z_q,
            "indices": indices,
            "dists": dists,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "active_num": active_num.float(),
        }

    def vq2emb(self, vq, out_proj=True):
        emb = self.embed_code(vq)
        if out_proj:
            emb = self.out_project(emb)
        return emb

    def tokenize(self, z: torch.Tensor) -> torch.Tensor:
        """tokenize the input tensor"""
        z_e = self.in_project(z)
        _, indices, _ = self.decode_latents(z_e)
        return indices

    def detokenize(self, indices):
        """detokenize the input indices"""
        z_q = self.decode_code(indices)
        z_q = self.out_project(z_q)
        return z_q

    def get_emb(self):
        return self.codebook.weight

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        # L2 normalize encodings and codebook
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance between encodings and codebook,
        # with L2 normalization, the distance is equal to cosine distance
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)

        return z_q, indices, dist


# =============================================================================
# >> END: PASTE CODE FROM sparktts/modules/* HERE <<
# =============================================================================


# =============================================================================
# >> START: PASTE CODE FROM sparktts/models/bicodec.py HERE <<
# =============================================================================
# IMPORTANT: The BiCodec class definition needs to be here.
# Modify its loading mechanism as suggested.


class BiCodec(nn.Module):
    def __init__(
        self,
        mel_params: Dict[str, Any],
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        speaker_encoder: nn.Module,
        prenet: nn.Module,
        postnet: nn.Module,
        **kwargs
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.speaker_encoder = speaker_encoder
        self.prenet = prenet
        self.postnet = postnet
        self.init_mel_transformer(mel_params)

    @classmethod
    def load_from_config_and_checkpoint(cls, model_dir: Path, bicodec_config_object: SparkTTSBiCodecConfig) -> "BiCodec":
        """
        Loads the BiCodec model using a SparkTTSBiCodecConfig object and a checkpoint file.
        Args:
            model_dir (Path): Path to the directory containing the model checkpoint ('model.safetensors').
            bicodec_config_object (SparkTTSBiCodecConfig): The nested config object from SparkTTSConfig.
        Returns:
            BiCodec: The initialized BiCodec model.
        """
        ckpt_path = model_dir / 'model.safetensors'
        if not ckpt_path.exists():
             ckpt_path_bin = model_dir / 'pytorch_model.bin'
             if ckpt_path_bin.exists():
                  ckpt_path = ckpt_path_bin
             else:
                 raise FileNotFoundError(f"BiCodec checkpoint not found at {model_dir / 'model.safetensors'} or potential fallbacks.")

        # Instantiate components using specific attributes from the nested config objects
        mel_params_config = bicodec_config_object.mel_params
        encoder_cfg = bicodec_config_object.encoder_config
        decoder_cfg = bicodec_config_object.decoder_config # WaveGenerator config
        quantizer_cfg = bicodec_config_object.quantizer_config
        speaker_encoder_cfg = bicodec_config_object.speaker_encoder_config
        prenet_cfg = bicodec_config_object.prenet_config
        postnet_cfg = bicodec_config_object.postnet_config

        # Pass only the arguments expected by each module's __init__
        mel_params = mel_params_config.to_dict() # Mel params might be needed as dict

        encoder = Encoder(
            input_channels=encoder_cfg.input_channels,
            vocos_dim=encoder_cfg.vocos_dim,
            vocos_intermediate_dim=encoder_cfg.vocos_intermediate_dim,
            vocos_num_layers=encoder_cfg.vocos_num_layers,
            out_channels=encoder_cfg.out_channels,
            sample_ratios=encoder_cfg.sample_ratios,
        )
        quantizer = FactorizedVectorQuantize(
            input_dim=quantizer_cfg.input_dim,
            codebook_size=quantizer_cfg.codebook_size,
            codebook_dim=quantizer_cfg.codebook_dim,
            commitment=quantizer_cfg.commitment,
            codebook_loss_weight=quantizer_cfg.codebook_loss_weight,
            decay=quantizer_cfg.decay,
            threshold_ema_dead_code=quantizer_cfg.threshold_ema_dead_code,
            # Add any other kwargs FactorizedVectorQuantize expects from its config
        )
        prenet = Decoder( # Assuming Prenet uses the Decoder class structure
            input_channels=prenet_cfg.input_channels,
            vocos_dim=prenet_cfg.vocos_dim,
            vocos_intermediate_dim=prenet_cfg.vocos_intermediate_dim,
            vocos_num_layers=prenet_cfg.vocos_num_layers,
            out_channels=prenet_cfg.out_channels,
            condition_dim=prenet_cfg.condition_dim,
            sample_ratios=prenet_cfg.sample_ratios,
            use_tanh_at_final=prenet_cfg.use_tanh_at_final,
        )
        postnet = Decoder( # Assuming Postnet uses the Decoder class structure
            input_channels=postnet_cfg.input_channels,
            vocos_dim=postnet_cfg.vocos_dim,
            vocos_intermediate_dim=postnet_cfg.vocos_intermediate_dim,
            vocos_num_layers=postnet_cfg.vocos_num_layers,
            out_channels=postnet_cfg.out_channels,
            # condition_dim=postnet_cfg.condition_dim, # Postnet might not have condition_dim
            # sample_ratios=postnet_cfg.sample_ratios, # Postnet might not have sample_ratios
            use_tanh_at_final=postnet_cfg.use_tanh_at_final,
        )
        decoder = WaveGenerator( # This is the actual audio decoder
            input_channel=decoder_cfg.input_channel,
            channels=decoder_cfg.channels,
            rates=decoder_cfg.rates,
            kernel_sizes=decoder_cfg.kernel_sizes,
            # d_out is likely fixed to 1 internally in WaveGenerator, not configured
        )
        speaker_encoder = SpeakerEncoder(
            input_dim=speaker_encoder_cfg.input_dim,
            out_dim=speaker_encoder_cfg.out_dim,
            latent_dim=speaker_encoder_cfg.latent_dim,
            token_num=speaker_encoder_cfg.token_num,
            fsq_levels=speaker_encoder_cfg.fsq_levels,
            fsq_num_quantizers=speaker_encoder_cfg.fsq_num_quantizers,
        )

        # Instantiate the BiCodec model itself
        model = cls(
            mel_params=mel_params, # Pass the dict here
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            speaker_encoder=speaker_encoder,
            prenet=prenet,
            postnet=postnet,
        )

        # --- State Dict Loading ---
        logger.info(f"Loading BiCodec state dict from: {ckpt_path}")
        if str(ckpt_path).endswith(".safetensors"):
            state_dict = load_file(ckpt_path, device="cpu") # Load to CPU first
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            logger.warning(f"BiCodec Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"BiCodec Unexpected keys: {unexpected_keys}")

        model.eval()
        model.remove_weight_norm() # Important step from original code

        logger.info("BiCodec loaded successfully.")
        return model
#
#     # --- Paste the rest of the BiCodec methods here ---
#     # forward, tokenize, detokenize, init_mel_transformer, remove_weight_norm

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a forward pass through the model.

        Args:
            batch (dict): A dictionary containing features, reference waveform, and target waveform.
        
        Returns:
            dict: A dictionary containing the reconstruction, features, and other metrics.
        """
        feat = batch["feat"]
        mel = self.mel_transformer(batch["ref_wav"]).squeeze(1)

        z = self.encoder(feat.transpose(1, 2))
        vq_outputs = self.quantizer(z)

        x_vector, d_vector = self.speaker_encoder(mel.transpose(1, 2))

        conditions = d_vector
        with_speaker_loss = False

        x = self.prenet(vq_outputs["z_q"], conditions)
        pred_feat = self.postnet(x)
        x = x + conditions.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return {
            "vq_loss": vq_outputs["vq_loss"],
            "perplexity": vq_outputs["perplexity"],
            "cluster_size": vq_outputs["active_num"],
            "recons": wav_recon,
            "pred_feat": pred_feat,
            "x_vector": x_vector,
            "d_vector": d_vector,
            "audios": batch["wav"].unsqueeze(1),
            "with_speaker_loss": with_speaker_loss,
        }


    @torch.no_grad()
    def tokenize(self, batch: Dict[str, Any]):
        """
        Tokenizes the input audio into semantic and global tokens.

        Args:
            batch (dict): The input audio features and reference waveform.

        Returns:
            tuple: Semantic tokens and global tokens.
        """
        feat = batch["feat"]
        mel = self.mel_transformer(batch["ref_wav"]).squeeze(1)
        z = self.encoder(feat.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)
        global_tokens = self.speaker_encoder.tokenize(mel.transpose(1, 2))

        return semantic_tokens, global_tokens

    @torch.no_grad()
    def detokenize(self, semantic_tokens, global_tokens):
        """
        Detokenizes the semantic and global tokens into a waveform.

        Args:
            semantic_tokens (tensor): Semantic tokens.
            global_tokens (tensor): Global tokens.

        Returns:
            tensor: Reconstructed waveform.
        """
        z_q = self.quantizer.detokenize(semantic_tokens)
        d_vector = self.speaker_encoder.detokenize(global_tokens)
        x = self.prenet(z_q, d_vector)
        x = x + d_vector.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return wav_recon

    def init_mel_transformer(self, config: Dict[str, Any]):
        """
        Initializes the MelSpectrogram transformer based on the provided configuration.

        Args:
            config (dict): Configuration parameters for MelSpectrogram.
        """
        import torchaudio.transforms as TT

        self.mel_transformer = TT.MelSpectrogram(
            config["sample_rate"],
            config["n_fft"],
            config["win_length"],
            config["hop_length"],
            config["mel_fmin"],
            config["mel_fmax"],
            n_mels=config["num_mels"],
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

    def remove_weight_norm(self):
        """Removes weight normalization from all layers."""
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                pass  # The module didn't have weight norm

        self.apply(_remove_weight_norm)

# =============================================================================
# >> END: PASTE CODE FROM sparktts/models/bicodec.py HERE <<
# =============================================================================


# =============================================================================
# >> START: PASTE CODE FROM sparktts/utils/audio.py HERE (if needed by model) <<
# =============================================================================
# Functions like audio_volume_normalize, load_audio, etc., are typically part
# of the Processor. However, if any are directly used *within* the BiCodec or
# other model components pasted above, they need to be defined here too.
# It seems `get_ref_clip` logic might be needed if `BiCodecTokenizer` logic is embedded.

# Example placeholder comment:

def audio_volume_normalize(audio: np.ndarray, coeff: float = 0.2) -> np.ndarray:
    """
    Normalize the volume of an audio signal.

    Parameters:
        audio (numpy array): Input audio signal array.
        coeff (float): Target coefficient for normalization, default is 0.2.

    Returns:
        numpy array: The volume-normalized audio signal.
    """
    # Sort the absolute values of the audio signal
    temp = np.sort(np.abs(audio))

    # If the maximum value is less than 0.1, scale the array to have a maximum of 0.1
    if temp[-1] < 0.1:
        scaling_factor = max(
            temp[-1], 1e-3
        )  # Prevent division by zero with a small constant
        audio = audio / scaling_factor * 0.1

    # Filter out values less than 0.01 from temp
    temp = temp[temp > 0.01]
    L = temp.shape[0]  # Length of the filtered array

    # If there are fewer than or equal to 10 significant values, return the audio without further processing
    if L <= 10:
        return audio

    # Compute the average of the top 10% to 1% of values in temp
    volume = np.mean(temp[int(0.9 * L) : int(0.99 * L)])

    # Normalize the audio to the target coefficient level, clamping the scale factor between 0.1 and 10
    audio = audio * np.clip(coeff / volume, a_min=0.1, a_max=10)

    # Ensure the maximum absolute value in the audio does not exceed 1
    max_value = np.max(np.abs(audio))
    if max_value > 1:
        audio = audio / max_value

    return audio


def load_audio(
    adfile: Path,
    sampling_rate: int = None,
    length: int = None,
    volume_normalize: bool = False,
    segment_duration: int = None,
) -> np.ndarray:
    r"""Load audio file with target sampling rate and lsength

    Args:
        adfile (Path): path to audio file.
        sampling_rate (int, optional): target sampling rate. Defaults to None.
        length (int, optional): target audio length. Defaults to None.
        volume_normalize (bool, optional): whether perform volume normalization. Defaults to False.
        segment_duration (int): random select a segment with duration of {segment_duration}s.
                                Defualt to None which means the whole audio will be used.

    Returns:
        audio (np.ndarray): audio
    """

    audio, sr = soundfile.read(adfile)
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    if sampling_rate is not None and sr != sampling_rate:
        audio = soxr.resample(audio, sr, sampling_rate, quality="VHQ")
        sr = sampling_rate

    if segment_duration is not None:
        seg_length = int(sr * segment_duration)
        audio = random_select_audio_segment(audio, seg_length)

    # Audio volume normalize
    if volume_normalize:
        audio = audio_volume_normalize(audio)
    # check the audio length
    if length is not None:
        assert abs(audio.shape[0] - length) < 1000
        if audio.shape[0] > length:
            audio = audio[:length]
        else:
            audio = np.pad(audio, (0, int(length - audio.shape[0])))
    return audio


def random_select_audio_segment(audio: np.ndarray, length: int) -> np.ndarray:
    """get an audio segment given the length

    Args:
        audio (np.ndarray):
        length (int): audio length = sampling_rate * duration
    """
    if audio.shape[0] < length:
        audio = np.pad(audio, (0, int(length - audio.shape[0])))
    start_index = random.randint(0, audio.shape[0] - length)
    end_index = int(start_index + length)

    return audio[start_index:end_index]


# =============================================================================
# >> END: PASTE CODE FROM sparktts/utils/audio.py HERE (if needed by model) <<
# =============================================================================


class SparkTTSModel(PreTrainedModel, GenerationMixin):
    """
    Spark-TTS model integrating a Language Model (LLM) for sequence generation,
    a Wav2Vec2 model for feature extraction, and a BiCodec model for audio
    tokenization and synthesis. Designed for compatibility with the Hugging Face ecosystem.
    """
    config_class = SparkTTSConfig
    base_model_prefix = "spark_tts" # Or perhaps "llm" if generation focuses there
    main_input_name = "input_ids" # Crucial for GenerationMixin

    def __init__(
        self,
        config: SparkTTSConfig,
        llm: Optional[PreTrainedModel] = None,
        wav2vec2_model: Optional[PreTrainedModel] = None,
        wav2vec2_processor: Optional[Wav2Vec2FeatureExtractor] = None, # Store processor too
        bicodec: Optional[nn.Module] = None, # Should be the loaded BiCodec instance
    ):
        super().__init__(config)
        self.config = config # Stores the main SparkTTSConfig

        # Store the sub-components
        self.llm = llm
        self.wav2vec2_model = wav2vec2_model
        self.wav2vec2_processor = wav2vec2_processor # Store the processor used for features
        self.bicodec = bicodec

        # Ensure Wav2Vec2 is configured for hidden states needed by BiCodec's feature extractor
        if self.wav2vec2_model:
             self.wav2vec2_model.config.output_hidden_states = True

        # Post initialization checks (optional but good practice)
        if not all([self.llm, self.wav2vec2_model, self.wav2vec2_processor, self.bicodec]):
            logger.warning(
                "SparkTTSModel initialized without all sub-components. "
                "Ensure `from_pretrained` is used for loading a complete model."
            )

    def get_input_embeddings(self):
        """Returns the input embeddings of the LLM."""
        if self.llm:
            return self.llm.get_input_embeddings()
        return None

    def set_input_embeddings(self, value):
        """Sets the input embeddings of the LLM."""
        if self.llm:
            self.llm.set_input_embeddings(value)

    def _prepare_wav2vec2_features(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Extracts Wav2Vec2 features required by BiCodec.
        Input wav should be a batch of waveforms [B, T_audio].
        """
        if not self.wav2vec2_model or not self.wav2vec2_processor:
            raise ValueError("Wav2Vec2 model or processor not loaded.")

        # Get target device and dtype from the Wav2Vec2 model
        target_device = self.wav2vec2_model.device
        target_dtype = self.wav2vec2_model.dtype # Get the model's dtype (e.g., bfloat16)

        # Input wav tensor might be float32, processor usually expects float32
        wav_for_processor = wav.to(device=target_device, dtype=torch.float32)

        # Process using the Wav2Vec2FeatureExtractor
        # The processor typically outputs float32
        inputs = self.wav2vec2_processor(
            wav_for_processor,
            sampling_rate=self.config.sample_rate, # Use config SR
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(target_device) # Move to device

        # --- Cast the input_values to the model's expected dtype ---
        input_values = input_values.to(dtype=target_dtype)
        # ----------------------------------------------------------

        # --- CRITICAL CHECK AND FIX ---
        # Ensure input_values is 2D [Batch, Length] before passing to the model
        if input_values.ndim == 3 and input_values.shape[1] == 1:
             logger.warning(f"Processor returned 3D input_values {input_values.shape}. Squeezing the channel dimension.")
             input_values = input_values.squeeze(1)
        elif input_values.ndim != 2:
             raise ValueError(f"Expected input_values from processor to be 2D [Batch, Length], but got shape {input_values.shape}")
        # --- END CHECK AND FIX ---

        # Extract features using the Wav2Vec2Model
        with torch.no_grad(): # Feature extraction should not require gradients here
            # Now the input dtype matches the model's parameter dtype
            feat_outputs = self.wav2vec2_model(input_values)

        # Combine specific hidden states as per original BiCodecTokenizer logic
        if not feat_outputs.hidden_states:
            raise ValueError("Wav2Vec2 model did not return hidden states. Ensure config.output_hidden_states=True.")
        if len(feat_outputs.hidden_states) < 17:
            # Wav2Vec2-large-xlsr has 24 layers + initial embeddings = 25 states
            logger.warning(f"Wav2Vec2 model returned {len(feat_outputs.hidden_states)} hidden states. Expected at least 17 for default BiCodec indices (11, 14, 16). Check model architecture or BiCodec indices if this is unexpected.")
            # Attempt to proceed if possible, otherwise raise error if indices are out of bounds
            idx1, idx2, idx3 = 11, 14, 16
            if not (0 <= idx1 < len(feat_outputs.hidden_states) and \
                    0 <= idx2 < len(feat_outputs.hidden_states) and \
                    0 <= idx3 < len(feat_outputs.hidden_states)):
                raise ValueError(f"Required hidden state indices ({idx1}, {idx2}, {idx3}) are out of bounds for the {len(feat_outputs.hidden_states)} hidden states returned.")
        else:
            idx1, idx2, idx3 = 11, 14, 16


        feats_mix = (
            feat_outputs.hidden_states[idx1] +
            feat_outputs.hidden_states[idx2] +
            feat_outputs.hidden_states[idx3]
        ) / 3

        # Ensure the output features also match the expected downstream dtype (e.g., bicodec)
        # Usually okay if subsequent layers also use the same target_dtype
        return feats_mix.to(dtype=target_dtype) # Return features in the target dtype # Shape: [B, T_feats, D_feats]
    
    @torch.no_grad()
    def tokenize_audio(self, wav: torch.Tensor, ref_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes audio using the BiCodec model.
        Args:
            wav (torch.Tensor): The main audio waveform [B, T_audio]. (Should be float32 initially)
            ref_wav (torch.Tensor): The reference audio waveform [B, T_ref_audio]. (Should be float32 initially)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: global_tokens, semantic_tokens
        """
        if not self.bicodec:
            raise ValueError("BiCodec model not loaded.")

        # 1. Extract Wav2Vec2 features for the main audio
        # _prepare_wav2vec2_features now handles internal dtype casting for w2v model
        feats = self._prepare_wav2vec2_features(wav) # Returns features in model's target dtype

        # 2. Prepare batch for BiCodec
        # Ensure tensors are on the BiCodec's device AND correct dtype
        # Get device and dtype from a BiCodec submodule parameter
        bicodec_param = next(self.bicodec.parameters())
        target_device = bicodec_param.device
        target_dtype = bicodec_param.dtype # Get BiCodec's dtype

        batch = {
            # Cast inputs to BiCodec's expected dtype
            "wav": wav.to(device=target_device, dtype=target_dtype),
            "ref_wav": ref_wav.to(device=target_device, dtype=target_dtype),
            "feat": feats.to(device=target_device, dtype=target_dtype), # Ensure feats are also correct dtype
        }

        # 3. Call BiCodec's tokenize method
        semantic_tokens, global_tokens = self.bicodec.tokenize(batch)

        return global_tokens, semantic_tokens
    
    @torch.no_grad()
    def detokenize_audio(self, global_tokens: torch.Tensor, semantic_tokens: torch.Tensor) -> np.ndarray:
        """
        Detokenizes audio tokens back to a waveform using BiCodec.
        Args:
            global_tokens (torch.Tensor): Global tokens [B, ...].
            semantic_tokens (torch.Tensor): Semantic tokens [B, ...].

        Returns:
            np.ndarray: The reconstructed waveform [T_audio_out] if B=1, or [B, T_audio_out] if B > 1,
                        with dtype float32 and values clipped to [-1, 1].
        """
        if not self.bicodec:
            raise ValueError("BiCodec model not loaded.")

        target_device = next(self.bicodec.parameters()).device

        # Adjust shapes as expected by BiCodec.detokenize if needed
        if global_tokens.ndim == 2: # Example adjustment
            global_tokens = global_tokens.unsqueeze(1)

        logger.debug(f"DEBUG: Detokenizing audio with global tokens {global_tokens.shape}, semantic tokens {semantic_tokens.shape}")

        wav_rec = self.bicodec.detokenize(
            semantic_tokens.to(target_device),
            global_tokens.to(target_device)
        ) # Output tensor likely float32 or model's dtype

        # Convert to numpy, ensure float32, clip
        wav_rec_np = wav_rec.detach().cpu().numpy().astype(np.float32) # Ensure float32
        wav_rec_np = np.clip(wav_rec_np, -1.0, 1.0) # Clip values

        logger.debug(f"DEBUG: Wav rec shape after detach and clip: {wav_rec_np.shape}") # Shape is likely (B, C, T) e.g., (1, 1, 24640)

        # ==============================================================
        # CORRECTED SQUEEZE LOGIC
        # ==============================================================
        # Remove all dimensions of size 1 (batch and channel if they are 1)
        # This handles both B=1, C=1 -> (T,) and potentially B>1, C=1 -> (B, T)
        # If C > 1, it would return (B, C, T) or (C, T) if B=1.
        # soundfile handles (T,) and (T, C) correctly.
        output_wav = wav_rec_np.squeeze()
        # ==============================================================

        logger.debug(f"DEBUG: Final output wav shape after squeeze: {output_wav.shape}")

        # Ensure the output is at least 1D even if squeeze removes everything (e.g., single sample output)
        if output_wav.ndim == 0:
            output_wav = np.expand_dims(output_wav, axis=0)

        return output_wav

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[list] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> dict:
        """
        Prepares inputs for the generation process (standard method for GenerationMixin).
        """
        # Add position_ids and handle past_key_values for causal LM generation
        # This is a standard implementation for causal LMs.
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            # Add any other inputs the LLM's forward method expects
        }

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # Add other potential inputs for the LLM (position_ids, past_key_values, etc.)
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        The forward pass primarily delegates to the underlying LLM.
        It takes tokenized text/audio prompts as input_ids.
        """
        if not self.llm:
            raise ValueError("LLM component not loaded.")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass arguments directly to the LLM's forward method
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs # Should be CausalLMOutputWithPast or tuple

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        # New args from base class signature to pass down if relevant
        state_dict = None,
        device_map = None,
        low_cpu_mem_usage = None,
        torch_dtype = "auto",
        quantization_config = None,
        trust_remote_code = None,
        # Add other relevant args from base class if needed: subfolder, variant, etc.
        subfolder: str = "", # Keep subfolder arg for overall loading logic
        variant: Optional[str] = None,
        **kwargs,
    ):
        # --- Argument Handling & Initial Setup ---
        if device_map:
            logger.warning("`device_map` is not directly supported for this composite model. Use .to(device) after loading.")
        if low_cpu_mem_usage:
             logger.info("`low_cpu_mem_usage` is set, but simplified loading is used. Memory usage might not be optimized.")
        if trust_remote_code is None:
             logger.warning("Loading SparkTTSModel requires custom code. Setting `trust_remote_code=True`.")
             trust_remote_code = True
        elif not trust_remote_code:
             raise ValueError("Loading SparkTTSModel requires `trust_remote_code=True`.")

        kwargs.pop("output_loading_info", None)
        kwargs.pop("_from_auto", None)
        kwargs.pop("attn_implementation", None)

        # --- 1. Resolve the main model directory ---
        if state_dict is not None:
             raise ValueError("Explicitly passing `state_dict` is not supported for this composite model.")
        if pretrained_model_name_or_path is None:
            raise ValueError("`pretrained_model_name_or_path` must be provided.")

        is_local = Path(pretrained_model_name_or_path).is_dir()
        if local_files_only and not is_local:
             raise ValueError(f"Cannot find local directory at {pretrained_model_name_or_path} when `local_files_only=True`.")

        if is_local:
            resolved_model_path = Path(pretrained_model_name_or_path)
            logger.info(f"Loading model from local directory: {resolved_model_path}")
        else:
            logger.info(f"{pretrained_model_name_or_path} is not a local directory. Assuming Hub ID and downloading.")
            try:
                # Use snapshot_download to get all necessary files
                # REMOVED subfolder=subfolder from this call
                resolved_model_path_str = snapshot_download(
                    repo_id=str(pretrained_model_name_or_path),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    allow_patterns=[
                        "*.json", "*.safetensors", "*.bin", "*.yaml", "*.txt",
                        "README.md", ".gitattributes",
                        "LLM/*", "BiCodec/*", "wav2vec2-large-xlsr-53/*"
                        ],
                    ignore_patterns=["*.git*", "*.h5", "*.ot", "*.msgpack"],
                    repo_type="model", # Explicitly set repo_type
                    # max_workers=..., # Can adjust workers if needed
                    # user_agent=..., # Can add user agent
                )
                resolved_model_path = Path(resolved_model_path_str)
                logger.info(f"Model files downloaded to cache: {resolved_model_path}")
            except Exception as e:
                # Catch potential TypeErrors from snapshot_download if args change again
                if isinstance(e, TypeError) and 'unexpected keyword argument' in str(e):
                     logger.error(f"snapshot_download() received an unexpected keyword argument. Check huggingface_hub version compatibility. Error: {e}")
                raise OSError(
                    f"Failed to download model '{pretrained_model_name_or_path}' (revision: '{revision}') from Hugging Face Hub. "
                    f"Error: {e}"
                )

        if not resolved_model_path.is_dir():
             raise EnvironmentError(f"Resolved model path is not a directory: {resolved_model_path}")

        # If subfolder was specified for from_pretrained, adjust the path *after* download
        if subfolder:
             resolved_model_path_with_subfolder = resolved_model_path / subfolder
             if not resolved_model_path_with_subfolder.is_dir():
                  raise EnvironmentError(f"Subfolder '{subfolder}' not found within the resolved path: {resolved_model_path}")
             resolved_model_path = resolved_model_path_with_subfolder # Update path to include subfolder
             logger.info(f"Using subfolder within resolved path: {resolved_model_path}")


        # --- 2. Load the main configuration ---
        if not isinstance(config, PretrainedConfig):
            # Load config from the potentially subfolder-adjusted path
            config_path = config if config is not None else resolved_model_path
            try:
                loaded_config, model_kwargs = SparkTTSConfig.from_pretrained(
                    config_path, # Load from the final resolved path
                    *model_args,
                    cache_dir=cache_dir,
                    force_download=force_download if not is_local else False,
                    local_files_only=local_files_only or is_local,
                    token=token,
                    revision=revision, # Pass revision for config loading too
                    trust_remote_code=trust_remote_code,
                    subfolder="", # Config is expected at the root of resolved_model_path
                    return_unused_kwargs=True,
                    **kwargs,
                )
                config = loaded_config
                kwargs = model_kwargs
            except OSError as e:
                 raise OSError(f"Cannot load config from {config_path}. Check `config.json` exists and is correctly formatted. Error: {e}")

        # --- Determine final torch_dtype ---
        final_torch_dtype = torch_dtype
        if final_torch_dtype == "auto":
            final_torch_dtype = getattr(config, "torch_dtype", None)
        if isinstance(final_torch_dtype, str) and final_torch_dtype != "auto":
            try:
                final_torch_dtype = getattr(torch, final_torch_dtype)
            except AttributeError:
                logger.warning(f"Invalid torch_dtype string: {final_torch_dtype}. Falling back to default.")
                final_torch_dtype = None
        elif final_torch_dtype == "auto":
             final_torch_dtype = None

        # --- Helper function to resolve component paths relative to the final resolved_model_path ---
        def _resolve_sub_path(sub_path_str):
            p = Path(sub_path_str)
            if p.is_absolute():
                if not p.exists(): logger.warning(f"Absolute path specified for sub-component does not exist: {p}")
                return str(p)
            else:
                # Resolve relative to the potentially subfolder-adjusted main model path
                resolved = resolved_model_path / p
                if not resolved.exists():
                     resolved_alt = resolved_model_path / sub_path_str.lstrip('./')
                     if resolved_alt.exists():
                          resolved = resolved_alt
                     else:
                          raise FileNotFoundError(f"Could not resolve sub-component path: {resolved} (relative to {resolved_model_path})")
                return str(resolved)

        # --- Component Loading Arguments ---
        component_loading_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision, # Pass revision to component loaders
            "trust_remote_code": trust_remote_code,
            "torch_dtype": final_torch_dtype,
            "use_safetensors": use_safetensors,
            "quantization_config": quantization_config if quantization_config else None,
            "variant": variant,
            **kwargs, # Pass remaining kwargs
        }

        # --- 3. Load Sub-components ---
        # (LLM, Wav2Vec2, BiCodec loading logic remains the same as previous version)
        # --- Load LLM ---
        llm_path = _resolve_sub_path(config.llm_model_name_or_path)
        logger.info(f"Loading LLM from resolved path: {llm_path}")
        try:
            # Pass subfolder="" because llm_path is now absolute or correctly relative
            llm = AutoModelForCausalLM.from_pretrained(
                llm_path, subfolder="", **component_loading_kwargs
            )
        except Exception as e:
            raise OSError(f"Failed to load LLM from {llm_path}: {e}")

        # --- Load Wav2Vec2 ---
        w2v_path = _resolve_sub_path(config.wav2vec2_model_name_or_path)
        logger.info(f"Loading Wav2Vec2 components from resolved path: {w2v_path}")
        try:
            # Load extractor without full component_loading_kwargs if they cause issues
            wav2vec2_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                w2v_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder="", # Path is resolved
            )
            # Load model with full kwargs
            wav2vec2_model = Wav2Vec2Model.from_pretrained(
                w2v_path, subfolder="", **component_loading_kwargs
            )
            wav2vec2_model.config.output_hidden_states = True
        except Exception as e:
            raise OSError(f"Failed to load Wav2Vec2 components from {w2v_path}: {e}")

        # --- Load BiCodec ---
        bicodec_path = _resolve_sub_path(config.bicodec_model_name_or_path)
        logger.info(f"Loading BiCodec from resolved path: {bicodec_path}")
        if not config.bicodec_config:
            raise ValueError("BiCodec configuration (`bicodec_config`) not found in SparkTTSConfig.")
        try:
            bicodec = BiCodec.load_from_config_and_checkpoint(
                model_dir=Path(bicodec_path),
                bicodec_config_object=config.bicodec_config
            )
            if not isinstance(bicodec, torch.nn.Module):
                 logger.warning("Loaded BiCodec component is not an instance of torch.nn.Module.")
            if isinstance(bicodec, torch.nn.Module) and final_torch_dtype:
                 bicodec = bicodec.to(dtype=final_torch_dtype)
        except FileNotFoundError as e:
             raise OSError(f"Failed to load BiCodec: Required file not found in {bicodec_path}. Error: {e}")
        except Exception as e:
             logger.error(f"Raw error loading BiCodec: {type(e).__name__}: {e}")
             import traceback
             traceback.print_exc()
             raise OSError(f"Failed to load BiCodec from {bicodec_path}. Error: {e}")


        # --- 4. Instantiate the main model wrapper ---
        model = cls(
            config,
            llm=llm,
            wav2vec2_model=wav2vec2_model,
            wav2vec2_processor=wav2vec2_processor,
            bicodec=bicodec
        )

        # --- 5. Handle device placement (Simplified) ---
        if torch.cuda.is_available():
             final_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
             final_device = torch.device("mps")
        else:
             final_device = torch.device("cpu")
        logger.info(f"Placing SparkTTSModel and components on device: {final_device}")
        try:
             model.to(final_device)
        except Exception as e:
             logger.error(f"Failed to move model to device {final_device}. Error: {e}")

        # --- 6. Return the loaded and prepared model ---
        return model