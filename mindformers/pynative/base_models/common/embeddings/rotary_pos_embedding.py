# Copyright 2026 Huawei Technologies Co., Ltd
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
]

import numpy as np
import mindspore as ms
from mindspore import nn, mint


class RotaryEmbedding(nn.Cell):
    """Rotary embedding for language model.

    Args:
        kv_channels (int): The number of channels for key and value.
        rotary_percent (float): The percentage of the rotary embedding.
        rotary_interleaved (bool): Whether to use interleaved rotary position embedding.
        seq_len_interpolation_factor (float): The interpolation factor for sequence length.
        rotary_base (int): The base for rotary embedding.
        rope_scaling (bool): Whether to enable dynamic RoPE scaling.
        rope_scaling_factor (float): The scaling factor used in RoPE scaling.
        use_eod_reset (bool): Whether to reset the positional offset at eod tokens.
    """

    def __init__(self,
                 kv_channels: int,
                 rotary_percent: float = 1.0,
                 rotary_interleaved: bool = False,
                 seq_len_interpolation_factor: float = None,
                 rotary_base: int = 10000,
                 rope_scaling: bool = False,
                 rope_scaling_factor: float = 8.0,
                 use_eod_reset: bool = False
                 ):
        super().__init__()
        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.mscale = 1.0
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.rotary_interleaved = rotary_interleaved
        self.use_eod_reset = use_eod_reset

        self.inv_freq = 1.0 / (rotary_base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        if rope_scaling:
            self.inv_freq = self._apply_scaling(self.inv_freq, factor=rope_scaling_factor)
        self.inv_freq = ms.Tensor(self.inv_freq, dtype=ms.float32)

        if self.rotary_interleaved:
            self.stack = mint.stack
        else:
            self.cat = mint.cat
        self.arange = mint.arange
        self.reshape = mint.reshape
        self.outer = mint.outer
        self.transpose = mint.transpose

    def _apply_scaling(
            self,
            freqs,
            factor=8,
            low_freq_factor=1,
            high_freq_factor=4,
            original_max_position_embeddings=8192,
    ):
        """apply rope scaling."""
        old_context_len = original_max_position_embeddings
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * np.pi / freqs
        inv_freq_llama = np.where(wavelen > low_freq_wavelen, freqs / factor, freqs)

        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_factor = np.clip(smooth_factor, 0.0, 1.0)

        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq_llama / factor) + smooth_factor * inv_freq_llama

        is_medium_freq = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)

        inv_freq_llama = np.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        return inv_freq_llama

    def construct(self, max_seq_len: int, offset: int = 0, position_ids=None):
        """Generate rotary position embedding.

        Args:
            max_seq_len (int): The maximum sequence length.
            offset (int): The offset for the sequence.
            position_ids: user self-defined position_ids for eod_reset.

        Returns:
            Tensor: Embeddings after applying RoPE.
            Tensor: mscale, return to match yarn interface.
        """
        if position_ids is None or not self.use_eod_reset:
            bs = 1
            seq = self.arange(max_seq_len, dtype=self.inv_freq.dtype) + offset
        else:
            bs = position_ids.shape[0]
            seq = self.reshape(position_ids, (-1,)) + offset

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = self.outer(seq, self.inv_freq)

        if self.rotary_interleaved:
            freqs_new_shape = (freqs.shape[0], -1)
            emb = self.reshape(self.stack([self.reshape(freqs, (-1, 1)), self.reshape(freqs, (-1, 1))], dim=-1),
                               freqs_new_shape)
        else:
            emb = self.cat((freqs, freqs), dim=-1)

        if position_ids is None or not self.use_eod_reset:
            out = self.reshape(emb, (-1, bs, 1, emb.shape[1]))
        else:
            out = self.reshape(emb, (bs, -1, 1, emb.shape[1]))
            out = self.transpose(out, 0, 1)
        return out, self.mscale
