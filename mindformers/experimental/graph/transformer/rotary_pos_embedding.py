# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
import math

from mindspore import nn, Tensor, dtype, mint
from mindspore import ops
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.ops.auto_generate import (AddExt, Reshape, Mul, Cos, Sin,
                                         Split, Neg, Concat, StackExt, StridedSlice, Div, Outer)
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig

__all__ = [
    "RotaryEmbedding",
    "ApplyRotaryPosEmb",
    "apply_rotary_pos_emb"
]


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
        """apply rope scaling"""
        # This implementation is adapted from:
        # https://github.com/huggingface/transformers/blob/2a5a6ad18aa22e98429bb5ecb880660328030ea0/src/transformers/modeling_rope_utils.py#L303-L343

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


class ApplyRotaryPosEmb(nn.Cell):
    """Apply rotary positional embedding to input tensor T.

    Args:
        config (TransformerConfig): The transformer configuration
    """
    def __init__(self,
                 config: TransformerConfig):
        super(ApplyRotaryPosEmb, self).__init__()
        self.add = AddExt()
        self.mul = Mul()
        self.cos = Cos()
        self.sin = Sin()
        self.neg = Neg()
        self.split = Split(axis=-1, output_num=2)
        self.cat = Concat(axis=-1)
        self.stack = StackExt(dim=-1)
        self.slice = StridedSlice()
        self.reshape = Reshape()

        if config is not None:
            if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
                self.sharding_propagation(config)
            else:
                self.shard(config)

    def construct(self, t: Tensor, freqs: Tensor, rotary_interleaved: bool = False) -> Tensor:
        """Apply rotary position embedding to input tensor.

        Args:
            t (Tensor): Input tensor is of shape [bs, n_heads, seq_len, head_dim]
            freqs (Tensor): Rotary position embedding frequencies of shape [1, 1, seq_len, head_dim]
            rotary_interleaved (bool): Whether to use interleaved rotary position embedding

        Returns:
            Tensor: Output tensor after applying rotary position embedding
        """
        if t.shape[-1] < freqs.shape[-1]:
            raise ValueError(
                f"Input tensor shape {t.shape} must be greater than freqs shape {freqs.shape} in the last dimension")
        if t.shape[-1] > freqs.shape[-1]:
            rot_dim = freqs.shape[-1]
            t, t_not_rotary = t[..., :rot_dim], t[..., rot_dim:]
        else:
            t_not_rotary = None

        cos_ = self.cos(freqs).astype(t.dtype)
        sin_ = self.sin(freqs).astype(t.dtype)

        output = self.add(self.mul(t, cos_), self.mul(self._rotate_half(t, rotary_interleaved), sin_))
        if t_not_rotary is not None:
            output = self.cat((output, t_not_rotary))
        return output

    def _rotate_half(self, t: Tensor, rotary_interleaved: bool = False) -> Tensor:
        bs, n_heads, seq_len, head_dim = t.shape
        if rotary_interleaved:
            t_1 = self.slice(t, (0, 0, 0, 0), (bs, n_heads, seq_len, head_dim), (1, 1, 1, 2))
            t_2 = self.slice(t, (0, 0, 0, 1), (bs, n_heads, seq_len, head_dim), (1, 1, 1, 2))
            t_rot = self.reshape(self.stack((self.neg(t_2), t_1)), (bs, n_heads, seq_len, -1))
        else:
            t_1, t_2 = self.split(t)
            t_rot = self.cat((self.neg(t_2), t_1))
        return t_rot

    def shard(self, config: TransformerConfig):
        """The multi-head attention naturally supports tensor parallelism by splitting along the head dimension."""
        dp = config.data_parallel if config and config.data_parallel is not None else 1
        tp = config.tensor_parallel if config and config.tensor_parallel is not None else 1

        sin_in_strategy = ((1, 1, 1, 1),)
        cos_in_strategy = ((1, 1, 1, 1),)
        split_in_strategy = ((dp, tp, 1, 1),)
        neg_in_strategy = ((dp, tp, 1, 1),)
        cat_in_strategy = ((dp, tp, 1, 1), (dp, tp, 1, 1))
        stack_in_strategy = ((dp, tp, 1, 1), (dp, tp, 1, 1))
        slice_in_strategy = ((dp, tp, 1, 1), (dp, tp, 1, 1))
        add_in_strategy = ((dp, tp, 1, 1), (dp, tp, 1, 1))
        mul_in_strategy = ((dp, tp, 1, 1), (1, 1, 1, 1))

        self.add.shard(in_strategy=add_in_strategy)
        self.mul.shard(in_strategy=mul_in_strategy)
        self.split.shard(in_strategy=split_in_strategy)
        self.cat.shard(in_strategy=cat_in_strategy)
        self.stack.shard(in_strategy=stack_in_strategy)
        self.slice.shard(in_strategy=slice_in_strategy)
        self.neg.shard(in_strategy=neg_in_strategy)
        self.sin.shard(in_strategy=sin_in_strategy)
        self.cos.shard(in_strategy=cos_in_strategy)

    def sharding_propagation(self, config: TransformerConfig):
        """The multi-head attention naturally supports tensor parallelism by splitting along the head dimension."""
        dp = config.data_parallel if config and config.data_parallel is not None else 1
        tp = config.tensor_parallel if config and config.tensor_parallel is not None else 1

        add_in_strategy = ((dp, tp, 1, 1), (dp, tp, 1, 1))
        mul_in_strategy = ((dp, tp, 1, 1), (1, 1, 1, 1))
        split_in_strategy = ((dp, tp, 1, 1),)

        self.add.shard(in_strategy=add_in_strategy)
        self.mul.shard(in_strategy=mul_in_strategy)
        self.split.shard(in_strategy=split_in_strategy)


def apply_rotary_pos_emb(t: Tensor, freqs: Tensor, config: TransformerConfig, cu_seqlens: Tensor = None) -> Tensor:
    if cu_seqlens is not None:
        raise NotImplementedError("For apply_rotary_pos_emb, cu_seqlens is not supported for now.")
    rotary_interleaved = config.rotary_interleaved if hasattr(config, 'rotary_interleaved') else False
    apply_func = ApplyRotaryPosEmb(config)
    return apply_func(t, freqs, rotary_interleaved)
