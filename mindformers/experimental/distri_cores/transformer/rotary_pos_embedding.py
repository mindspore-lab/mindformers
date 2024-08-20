# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Rotary position embedding."""
import numpy as np

from mindspore import Tensor, ops, mint

from mindformers.experimental.distri_cores.transformer.module import Module


__all__ = ["RotaryEmbedding", "apply_rotary_pos_emb"]


class RotaryEmbedding(Module):
    r"""
    Rotary positional embedding for language model.

    Args:
        head_dim (int): Per head hidden_size.
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings. Default: 1.0.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Default: None.
        rotary_base (int): Base period for rotary position embeddings. Default: 10000.

    Inputs:
        - **max_seq_len** (int) - Max sequence length of inputs.
        - **offset** (int) - Offset.

    Outputs:
        - **emb** (Tensor) - The input tensor after applying RoPE.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=False,
            seq_len_interpolation_factor=None,
            rotary_base=10000,
        ):
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved
        if self.rotary_interleaved:
            raise NotImplementedError('Rotary interleaved is not supported for now.')

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            rotary_base ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim)
        )

    def construct(self, max_seq_len, offset=0):
        """ Construct function of rotary embedding. """
        seq = (np.arange(max_seq_len, dtype=self.inv_freq.dtype) + offset).astype(np.float32)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = np.outer(seq, self.inv_freq)

        if not self.rotary_interleaved:
            emb = np.concatenate((freqs, freqs), axis=-1)
        else:
            raise NotImplementedError('Rotary interleaved is not supported for now.')

        # emb [.., S, D]
        emb = emb[np.newaxis, np.newaxis, :, :]
        return Tensor(emb)


def apply_rotary_pos_emb(t, freqs, config=None, cu_seqlens=None) -> Tensor:
    """
    Apply rotary positional embedding to input tensor.
    Please check https://kexue.fm/archives/8265 for detailed formulas

    Inputs:
        - **x** (Tensor) - Input tensor x with shape :math:`(B, N, S, D)`.
        - **freqs** (Tensor) - Rotary positional embedding tensor freq with shape :math:`(..., S, D)`.

    Outputs:
        - **output** (Tensor): The input tensor after applying RoPE.

    Supported Platforms:
        ``Ascend``
    """
    if config is not None:
        raise NotImplementedError('config input for apply_rotary_pos_emb() is not supported for now.')
    if cu_seqlens is not None:
        raise NotImplementedError('cu_seqlens input for apply_rotary_pos_emb() is not supported for now.')

    cos_ = mint.cos(freqs).astype(t.dtype)
    sin_ = mint.sin(freqs).astype(t.dtype)

    # rotate
    t_splited = mint.split(t, t.shape[-1] // 2, dim=-1)
    t_1 = t_splited[0]
    t_2 = t_splited[1]
    t_rotate = ops.cat((-t_2, t_1), axis=-1)

    output = (t * cos_) + (t_rotate * sin_)
    return output
