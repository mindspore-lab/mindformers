# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Rotary position embedding for transformer.
"""
__all__ = [
    "RotaryEmbedding",
    "YarnRotaryEmbedding",
]

import math

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, dtype, mint
from mindspore import ops
from mindspore.ops.auto_generate import (Reshape, Mul, Concat, StackExt, Div, Outer)


class RotaryEmbedding(nn.Cell):
    """Rotary embedding for language model.

    Args:
        kv_channels (int): The number of channels for key and value.
        rotary_percent (float): The percentage of the rotary embedding.
        seq_len_interpolation_factor (float): The interpolation factor for sequence length.
        rotary_base (int): The base for rotary embedding.
        rotary_interleaved (bool): Whether to use interleaved rotary position embedding.
    """

    def __init__(self,
                 kv_channels: int,
                 rotary_percent: float = 1.0,
                 rotary_interleaved: bool = False,
                 seq_len_interpolation_factor: float = None,
                 rotary_base: int = 10000,
                 rope_scaling: bool = False,
                 rope_scaling_factor: float = 8.0,
                 use_cpu_initialization: bool = False,
                 ):
        super(RotaryEmbedding, self).__init__()
        if use_cpu_initialization:
            raise NotImplementedError("For RotaryEmbedding, "
                                      "use_cpu_initialization is not supported for now.")
        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.mscale = 1.0
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.rotary_interleaved = rotary_interleaved

        self.mul = Mul()
        self.div = Div()
        self.inv_freq = self.div(1.0, (
            rotary_base ** (ops.arange(0, dim, 2, dtype=dtype.float32) / dim)
        ))

        if rope_scaling:
            self.inv_freq = self._apply_scaling(self.inv_freq, factor=rope_scaling_factor)

        if self.rotary_interleaved:
            self.stack = StackExt(dim=-1)
        else:
            self.cat = Concat(axis=-1)
        self.reshape = Reshape()

        self.outer = Outer()

    def _apply_scaling(self,
                       freqs,
                       factor=8,
                       low_freq_factor=1,
                       high_freq_factor=4,
                       original_max_position_embeddings=8192,
                       ):
        # This implementation is adapted from:
        # https://github.com/huggingface/transformers/blob/2a5a6ad18aa22e98429bb5ecb880660328030ea0/src/transformers/modeling_rope_utils.py#L303-L343
        """apply rope scaling"""

        factor = factor  # `8` in the original implementation
        low_freq_factor = low_freq_factor  # `1` in the original implementation
        high_freq_factor = high_freq_factor  # `4` in the original implementation
        old_context_len = original_max_position_embeddings  # `8192` in the original implementation

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / freqs
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = mint.where(wavelen > low_freq_wavelen, freqs / factor, freqs)

        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = mint.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        return inv_freq_llama

    def construct(self, max_seq_len: int, offset: int = 0, position_ids=None):
        """Generate rotary position embedding.

        Args:
            max_seq_len (int): The maximum sequence length.
            offset (int): The offset for the sequence.

        Returns:
            Tensor: Embeddings after applying RoPE.
            Tensor: mscale, return to match yarn interface.
        """
        if position_ids is None:
            seq = ops.arange(max_seq_len, dtype=self.inv_freq.dtype) + offset
        else:
            seq = position_ids + offset

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = self.outer(seq, self.inv_freq)

        if self.rotary_interleaved:
            freqs_new_shape = (freqs.shape[0], -1)
            emb = self.reshape(self.stack([self.reshape(freqs, (-1, 1)), self.reshape(freqs, (-1, 1))]),
                               freqs_new_shape)
        else:
            emb = self.cat((freqs, freqs))

        # emb[seq_length, .., dim]
        out = self.reshape(emb, (emb.shape[0], -1, 1, emb.shape[1]))
        return out.copy(), self.mscale


class YarnRotaryEmbedding(RotaryEmbedding):
    """Yarn Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained from
            transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for
            longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (float, optional): Base period for rotary position embeddings. Defaults to
            10000.
        use_cpu_initialization (bool, optional): If False, initialize the inv_freq directly on
            the NPU. Defaults to False
        scaling_factor (float, optional): Scaling factor for Yarn RoPE. Defaults to 1.0.
        original_max_position_embeddings (int, optional): Original maximum position embeddings
            length. Defaults to 4096.
        beta_fast (float, optional): Fast beta value for Yarn RoPE. Defaults to 32.
        beta_slow (float, optional): Slow beta value for Yarn RoPE. Defaults to 1.
        mscale (float, optional): Mscale value for Yarn RoPE. Defaults to 1.
        mscale_all_dim (float, optional): Mscale all dim value for Yarn RoPE. Defaults to 0.
    """

    def __init__(self,
                 kv_channels: int,
                 rotary_percent: float = 1.0,
                 rotary_interleaved: bool = False,
                 seq_len_interpolation_factor: float = None,
                 rotary_base: float = 10000.0,
                 use_cpu_initialization: bool = False,
                 scaling_factor: float = 1.0,
                 original_max_position_embeddings: int = 4096,
                 beta_fast: float = 32.0,
                 beta_slow: float = 1.0,
                 mscale: float = 1.0,
                 mscale_all_dim: float = 0.0,
                 ):
        internal_freq_base = np.arange(0, kv_channels, 2)[: (kv_channels // 2)].astype(np.float32)
        internal_freq = 1.0 / (scaling_factor * rotary_base ** (internal_freq_base / kv_channels))

        extra_freq_base = np.arange(0, kv_channels, 2)[: (kv_channels // 2)].astype(np.float32)
        extra_freq = 1.0 / (rotary_base ** (extra_freq_base / kv_channels))

        low, high = _yarn_find_correction_range(beta_fast, beta_slow, kv_channels, rotary_base,
                                                original_max_position_embeddings)
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, kv_channels // 2)
        freqs = internal_freq * (1 - inv_freq_mask) + extra_freq * inv_freq_mask

        self.mscale = float(_yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(scaling_factor, mscale_all_dim))
        self.freqs = Tensor(freqs, dtype=ms.float32)

        super().__init__(
            kv_channels,
            rotary_percent,
            rotary_interleaved,
            seq_len_interpolation_factor,
            rotary_base,
            use_cpu_initialization,
        )

    def construct(self, max_seq_len: int, offset: int = 0, position_ids=None):
        """Generate rotary position embedding.

        Args:
            max_seq_len (int): The maximum sequence length.
            offset (int): The offset for the sequence.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        if position_ids is None:
            seq = ops.arange(max_seq_len, dtype=self.inv_freq.dtype) + offset
        else:
            seq = position_ids + offset

        freqs = self.outer(seq, self.freqs)

        emb = self.cat((freqs, freqs))

        # emb[seq_length, .., dim]
        out = self.reshape(emb, (emb.shape[0], -1, 1, emb.shape[1]))
        return out.copy(), self.mscale


def get_swap_mask(head_dim):
    """Swap matrix"""
    zero_block = np.zeros((head_dim // 2, head_dim // 2), dtype=np.float32)
    id_block = np.identity(head_dim // 2, dtype=np.float32)
    return np.block([[zero_block, id_block], [-id_block, zero_block]])


def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    """Inverse dim formula to find dim based on number of rotations"""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    """Find dim range bounds based on rotations"""
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min_, max_, dim):
    if min_ == max_:
        max_ += 0.001  # Prevent singularity

    linear_func = (np.arange(dim, dtype=np.float32) - min_) / (max_ - min_)
    ramp_func = np.clip(linear_func, 0, 1, out=None)
    return ramp_func


def _yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
