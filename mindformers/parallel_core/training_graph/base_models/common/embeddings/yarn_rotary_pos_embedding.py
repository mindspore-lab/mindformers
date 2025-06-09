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
__all__ = [
    "YarnRotaryEmbedding"
]

import math

import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindformers.parallel_core.training_graph.base_models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding)


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
                 use_eod_reset: bool = False
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
            use_eod_reset
        )

    def construct(self, max_seq_len: int, offset: int = 0, position_ids=None):
        """Generate rotary position embedding.

        Args:
            max_seq_len (int): The maximum sequence length.
            offset (int): The offset for the sequence.
            position_ids: user self-defined position_ids for eod_reset.

        Returns:
            Tensor: Embeddings after applying RoPE.
            Tensor: mscale.
        """
        if position_ids is None or not self.use_eod_reset:
            bs = 1
            seq = ops.arange(max_seq_len, dtype=self.inv_freq.dtype) + offset
            outer = self.outer
            cat = self.cat
        else:
            bs = position_ids.shape[0]
            seq = self.reshape(position_ids, (-1,)) + offset
            outer = self.outer_for_eod
            cat = self.cat_for_eod

        freqs = outer(seq, self.freqs)

        emb = cat((freqs, freqs))

        if position_ids is None or not self.use_eod_reset:
            # emb[seq_length, .., dim]
            out = self.reshape(emb, (-1, bs, 1, emb.shape[1]))
        else:
            out = self.reshape(emb, (bs, -1, 1, emb.shape[1]))
            out = self.transpose(out, (1, 0, 2, 3))
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
