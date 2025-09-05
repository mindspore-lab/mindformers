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
Apply rotary position embedding for transformer.
"""
__all__ = [
    "ApplyRotaryPosEmb"
]

from mindspore import nn, Tensor, ops
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.ops.auto_generate import AddExt, Reshape, Mul, Cos, Sin, Split, Neg, Concat, StackExt, StridedSlice
from mindformers.parallel_core.training_graph.device_matrix import layout
from mindformers.parallel_core.transformer_config import TransformerConfig, MLATransformerConfig
from mindformers.parallel_core.training_graph.base_models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding)
from mindformers.parallel_core.training_graph.base_models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding)


def _get_rope(config: MLATransformerConfig):
    return RotaryEmbedding(
        config.qk_pos_emb_head_dim,
        rotary_percent=config.rotary_percent,
        rotary_interleaved=config.rotary_interleaved,
        seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
        rotary_base=config.rotary_base,
        rope_scaling=config.use_rope_scaling
    )


def _get_yarn(config: MLATransformerConfig):
    return YarnRotaryEmbedding(
        config.qk_pos_emb_head_dim,
        rotary_percent=config.rotary_percent,
        rotary_interleaved=config.rotary_interleaved,
        seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
        rotary_base=config.rotary_base,
        scaling_factor=config.rotary_scaling_factor,
        original_max_position_embeddings=config.max_position_embeddings,
        beta_fast=config.beta_fast,
        beta_slow=config.beta_slow,
        mscale=config.mscale,
        mscale_all_dim=config.mscale_all_dim,
    )


ROPE_FUNCTIONS = {
    'rope': _get_rope,
    'yarn': _get_yarn,
}


class ApplyRotaryPosEmb(nn.Cell):
    """Apply rotary positional embedding to input tensor T.

    Args:
        config (TransformerConfig): The transformer configuration
    """

    def __init__(self,
                 config: TransformerConfig,
                 for_k_pos_emb=False
                 ):
        super(ApplyRotaryPosEmb, self).__init__()
        self.append_eod = config.use_eod_reset
        self.add = AddExt()
        self.mul = Mul()
        self.mul_mscale = Mul()
        self.cos = Cos()
        self.sin = Sin()
        self.neg = Neg()
        self.split = Split(axis=-1, output_num=2)
        self.cat = Concat(axis=-1)
        self.stack = StackExt(dim=-1)
        self.slice = StridedSlice()
        self.strideslice = StridedSlice()
        self.reshape = Reshape()
        self.apply_rope_fusion = config.apply_rope_fusion
        self.for_k_pos_emb = for_k_pos_emb

        if self.apply_rope_fusion:
            self.rope = ops.auto_generate.gen_ops_prim.RotaryPositionEmbedding()

        if config is not None:
            if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
                self.sharding_propagation(config)
            elif _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
                self.shard(config)

    def construct(self,
                  t: Tensor,
                  freqs: tuple,
                  rotary_interleaved: bool = False,
                  multi_latent_attention: bool = False) -> Tensor:
        """Apply rotary position embedding to input tensor.

        Args:
            t (Tensor): Input tensor is of shape [seq_length, ... , dim]
            freqs (tuple): A tuple of  frequencies and mscale. Rotary position embedding frequencies
                of shape [seq_length, ... , dim], mscale is float
            rotary_interleaved (bool): Whether to use interleaved rotary position embedding. Default: False
            multi_latent_attention (bool): Whether to use multi latent attention. Default: False

        Returns:
            Tensor: Output tensor after applying rotary position embedding
        """
        seq_len, bs, n_heads, head_dim = t.shape
        freqs, m_scale = freqs
        rot_dim = freqs.shape[-1]
        if head_dim == rot_dim:
            t_not_rotary = None
        else:
            t_not_rotary = self.slice(t, (0, 0, 0, rot_dim), (seq_len, bs, n_heads, head_dim), (1, 1, 1, 1))
            t = self.slice(t, (0, 0, 0, 0), (seq_len, bs, n_heads, rot_dim), (1, 1, 1, 1))

        if multi_latent_attention:
            x1 = self.strideslice(t, (0, 0, 0, 0), (seq_len, bs, n_heads, head_dim), (1, 1, 1, 2))
            x2 = self.strideslice(t, (0, 0, 0, 1), (seq_len, bs, n_heads, head_dim), (1, 1, 1, 2))
            t = self.cat((x1, x2))

        cos_ = self.cos(self.mul_mscale(freqs, m_scale)).astype(t.dtype)
        sin_ = self.sin(self.mul_mscale(freqs, m_scale)).astype(t.dtype)

        if self.apply_rope_fusion:
            output = self.rope(t, cos_, sin_, 0)
        else:
            t_rot = self._rotate_half(t, rotary_interleaved)
            output = self.add(self.mul(t, cos_), self.mul(t_rot, sin_))

        if t_not_rotary is not None:
            output = self.cat((output, t_not_rotary))
        return output

    def _rotate_half(self, t: Tensor, rotary_interleaved: bool = False) -> Tensor:
        """Rotates half of the input tensor for rotary position embeddings.

        Args:
            t: Input tensor of shape (seq_len, bs, n_heads, head_dim)
            rotary_interleaved: If True, processes interleaved features

        Returns:
            Rotated tensor with same shape as input
        """
        seq_len, bs, n_heads, head_dim = t.shape
        if rotary_interleaved:
            t_1 = self.slice(t, (0, 0, 0, 0), (seq_len, bs, n_heads, head_dim), (1, 1, 1, 2))
            t_2 = self.slice(t, (0, 0, 0, 1), (seq_len, bs, n_heads, head_dim), (1, 1, 1, 2))
            t_rot = self.reshape(self.stack((self.neg(t_2), t_1)), (seq_len, bs, n_heads, -1))
        else:
            t_1, t_2 = self.split(t)
            t_rot = self.cat((self.neg(t_2), t_1))
        return t_rot

    def shard(self, config: TransformerConfig):
        """The multi-head attention naturally supports tensor parallelism by splitting along the head dimension."""
        self.cos.add_prim_attr("self_define_shard", True)
        self.sin.add_prim_attr("self_define_shard", True)
        self.slice.add_prim_attr("self_define_shard", True)
        self.strideslice.add_prim_attr("self_define_shard", True)
        self.cat.add_prim_attr("self_define_shard", True)

        self.mul_mscale.shard(in_strategy=(layout("None", "None", "None", "None"), layout("None",)))
        self.cos.shard(in_strategy=(layout("None", "None", "None", "None"),),
                       out_strategy=(layout("None", "None", "None", "None"),)
                       )
        self.sin.shard(in_strategy=(layout("None", "None", "None", "None"),),
                       out_strategy=(layout("None", "None", "None", "None"),)
                       )

        if not self.for_k_pos_emb:
            self.split.shard((layout("cp", "dp", "tp", "None"),))
            self.neg.shard((layout("cp", "dp", "tp", "None"),))
            self.add.shard((layout("cp", "dp", "tp", "None"), layout("cp", "dp", "tp", "None")))
            self.mul.shard(in_strategy=(layout("None", "dp", "tp", "None"), layout("None", "None", "None", "None")))
            self.slice.shard(in_strategy=(layout("None", "dp", "tp", "None"),),
                             out_strategy=(layout("None", "dp", "tp", "None"),))
            self.strideslice.shard(in_strategy=(layout("None", "dp", "tp", "None"),),
                                   out_strategy=(layout("None", "dp", "tp", "None"),))
            self.cat.shard(in_strategy=((layout("None", "dp", "tp", "None"), layout("None", "dp", "tp", "None")),),
                           out_strategy=(layout("None", "dp", "tp", "None"),))
        else:
            self.split.shard((layout("cp", "dp", "None", "None"),))
            self.neg.shard((layout("cp", "dp", "None", "None"),))
            self.add.shard((layout("cp", "dp", "None", "None"), layout("cp", "dp", "None", "None")))
            self.mul.shard(in_strategy=(layout("None", "dp", "None", "None"), layout("None", "None", "None", "None")))
            self.slice.shard(in_strategy=(layout("None", "dp", "None", "None"),),
                             out_strategy=(layout("None", "dp", "None", "None"),))
            self.strideslice.shard(in_strategy=(layout("None", "dp", "None", "None"),),
                                   out_strategy=(layout("None", "dp", "None", "None"),))
            self.cat.shard(in_strategy=((layout("None", "dp", "None", "None"), layout("None", "dp", "None", "None")),),
                           out_strategy=(layout("None", "dp", "None", "None"),))

        if self.apply_rope_fusion:
            self.rope.shard(in_strategy=(layout("cp", "dp", "tp", "None"),
                                         layout("cp", "None", "None", "None"),
                                         layout("cp", "None", "None", "None")),
                            out_strategy=(layout("cp", "dp", "tp", "None"),)
                            )
            self.rope.add_prim_attr("self_define_shard", True)


    def sharding_propagation(self, config: TransformerConfig):
        """The multi-head attention naturally supports tensor parallelism by splitting along the head dimension."""
        dp = config.data_parallel_size if config and config.data_parallel_size is not None else 1
        tp = config.tensor_model_parallel_size if config and config.tensor_model_parallel_size is not None else 1

        strategy_in = (1, dp, tp, 1)

        add_in_strategy = (strategy_in, strategy_in)
        split_in_strategy = (strategy_in,)

        self.add.shard(in_strategy=add_in_strategy)
        self.split.shard(in_strategy=split_in_strategy)
        if self.append_eod:
            self.mul.shard(in_strategy=(strategy_in, (1, dp, 1, 1)))
        else:
            self.mul.shard(in_strategy=(strategy_in, (1, 1, 1, 1)))
