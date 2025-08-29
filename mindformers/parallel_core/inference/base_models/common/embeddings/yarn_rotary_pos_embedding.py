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
Yarn rotary position embedding for transformer.
"""
__all__ = ['YaRNScalingRotaryEmbedding']

from typing import Tuple, Union
import math

from mindspore import ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor

from mindformers.parallel_core.inference.base_models.common.embeddings.rotary_pos_embedding import RotaryEmbedding


class YaRNScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Args:
        kv_channels (int): The number of channels for key and value.
        rotary_percent (float): The percentage of the rotary embedding.
        rotary_interleaved (bool): Whether to use interleaved rotary position embedding.
        seq_len_interpolation_factor (float): The interpolation factor for sequence length.
        rotary_base (float): The base for rotary embedding.
        rotary_cos_format (str): The mode of ApplyRotaryPosEmb.
        rotary_dtype (mstype): The dtype of rotary embeddings.
        scaling_factor (float): Base scaling factor for sequence length adjustment.
        original_max_position_embeddings (int): Original maximum position embeddings.
        beta_fast (int): High-frequency adjustment cutoff parameter. Larger values
                         extend high-frequency preservation. Default: 32.
        beta_slow (int): Low-frequency adjustment cutoff parameter. Default: 1.
        mscale (float): Primary magnitude scaling coefficient.
        mscale_all_dim (float): Full-dimensional magnitude scaling coefficient.
        extrapolation_factor (float): Weighting factor for extrapolated frequency ranges.
        attn_factor (float): Global attention stability coefficient.

    """
    def __init__(
            self,
            kv_channels: int,
            rotary_percent: float = 1.0,
            rotary_interleaved: bool = False,
            seq_len_interpolation_factor: float = None,
            rotary_base: float = 10000.0,
            rotary_cos_format: str = "rotate_half",
            rotary_dtype: mstype = mstype.float16,
            scaling_factor: float = 1.0,
            original_max_position_embeddings: int = 4096,
            beta_fast: float = 32.0,
            beta_slow: float = 1.0,
            mscale: float = 1.0,
            mscale_all_dim: float = 0.0,
            extrapolation_factor: float = 1.0,
            attn_factor: float = 1.0
    ) -> None:
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            _yarn_get_mscale(self.scaling_factor, float(mscale)) /
            _yarn_get_mscale(self.scaling_factor, float(mscale_all_dim)) *
            attn_factor)
        super().__init__(kv_channels, rotary_percent, rotary_interleaved, seq_len_interpolation_factor,
                         rotary_base, rotary_cos_format, rotary_dtype, original_max_position_embeddings)

    def _compute_inv_freq(self, base: Union[int, float]) -> Tensor:
        """Compute the inverse frequency."""
        pos_freqs = base**(
            ops.arange(0, self.dim, 2, dtype=mstype.float32)[: (self.dim // 2)] / self.dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                                self.dim, base, self.original_max_position_embeddings)
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(
            low, high, self.dim // 2,
            dtype=mstype.float32)) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (
            1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def get_freqs_non_repeated(self, offset: int = 0) -> Tensor:
        """
        Generates matrix of frequencies based on positions in the sequence,
        used to create positional encodings
        """
        inv_freq = self._compute_inv_freq(self.rotary_base)
        seq = ops.arange(0, self.max_position_embeddings * self.scaling_factor, 1, dtype=inv_freq.dtype) + offset

        freqs = ops.outer(seq, inv_freq)

        return freqs

    def _compute_cos_sin_cache(self) -> Tuple[Tensor, Tensor]:
        freqs = self.get_freqs_non_repeated()
        emb = self.cat((freqs, freqs))

        cos = self.cast(self.cos(emb) * self.mscale, self.rotary_dtype)
        sin = self.cast(self.sin(emb) * self.mscale, self.rotary_dtype)

        return cos, sin


def _yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    """Computes dynamic magnitude scaling coefficient based on scaling factor"""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(num_rotations: int,
                              dim: int,
                              base: float = 10000,
                              max_position_embeddings: int = 2048) -> float:
    """Inverse dim formula to find dim based on number of rotations"""
    return (dim * math.log(max_position_embeddings /
                           (num_rotations * 2 * math.pi))) / (2 *
                                                              math.log(base))


def _yarn_find_correction_range(
        low_rot: int,
        high_rot: int,
        dim: int,
        base: float = 10000,
        max_position_embeddings: int = 2048) -> Tuple[int, int]:
    """Find dim range bounds based on rotations"""
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base,
                                  max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(low: float, high: float, dim: int,
                           dtype: mstype) -> Tensor:
    """Generates a linear ramp mask for smooth transition."""
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (ops.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = ops.clamp(linear_func, 0, 1)
    return ramp_func
