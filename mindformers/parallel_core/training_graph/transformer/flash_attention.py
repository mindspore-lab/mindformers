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
"""Flash Attention Layer"""
__all__ = ['FlashAttention']

import math

import mindspore.common.dtype as mstype
import mindspore as ms
from mindspore import ops, ParallelMode
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.ops import auto_generate as aclnn_ops
from mindspore.ops import functional as F
from mindspore.ops import cast
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore.parallel.shard import Layout
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.parallel_core.training_graph.transformer.enums import AttnMaskType


class FlashAttention(Cell):
    """
    FlashAttention Layer.

    This class implements the FlashAttention mechanism for fast and memory-efficient attention computation.
    It supports multiple attention types, mask modes, and is optimized for parallel training including
    tensor and context parallelism.

    Reference:
        "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
        https://arxiv.org/abs/2205.14135

    Args:
        config (MLATransformerConfig): Configuration object containing model hyperparameters,
            including number of heads, dropout probabilities, and more.
        layer_number (int): The index of the current layer within the transformer stack.
        attn_mask_type (AttnMaskType, optional): Type of attention mask to apply (e.g., "causal", "padding").
            Default is None.
        attention_type (str, optional): Specifies the attention type, such as "self" or "cross".
            Default is None.
        attention_dropout (float, optional): Dropout probability applied to attention weights.
            If not provided, this is taken from the config.
        softmax_scale (float, optional): Scaling factor for the attention logits before softmax.
            If None, it defaults to 1 / sqrt(head_dim).
        cp_comm_type (str, optional): Type of communication for context parallelism.
            Default is None.

    Inputs:
        - **query** (Tensor): The query tensor with shape (B, S1, H1) or (B, N1, S1, D).
        - **key** (Tensor): The key tensor with shape (B, S2, H2) or (B, N2, S2, D).
        - **value** (Tensor): The value tensor with shape (B, S2, H2) or (B, N2, S2, D).
        - **attn_mask** (Tensor, optional): Attention mask. A value of 0 keeps the element;
          a value of 1 masks it out. Shape can vary based on attention mode.
        - **alibi_mask** (Tensor, optional): Positional bias tensor for ALiBi attention.
          Used for large sequences and causal masks.
        - **prefix** (Tensor, optional): Prefix lengths for prefix attention mode.
          Not implemented yet.
        - **padding_mask** (None): Reserved for future use.
        - **actual_seq_qlen** (Tensor[int32], optional): Actual valid sequence lengths of the query.
        - **actual_seq_kvlen** (Tensor[int32], optional): Actual valid sequence lengths of the key/value.

    Outputs:
        - **attention_out** (Tensor): The attention output tensor with the same shape and type as `query`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self,
                 config: MLATransformerConfig,
                 layer_number,
                 attn_mask_type: AttnMaskType = None,
                 attention_type: str = None,
                 attention_dropout: float = None,
                 softmax_scale: float = None,
                 cp_comm_type: str = None,
                 ):
        super(FlashAttention, self).__init__()

        # FA (Flash Attention) is an optimized version of DotProductAttention in Megatron v0.12.0,
        # with nearly identical computational precision.

        if attn_mask_type:
            raise NotImplementedError("For FlashAttention, 'attn_mask_type' is not supported for now.")
        if attention_type:
            raise NotImplementedError("For FlashAttention, 'attention_type' is unused for now.")
        if cp_comm_type:
            raise NotImplementedError("For FlashAttention, 'cp_comm_type' is not supported for now.")

        self.config = config
        self.layer_number = max(1, layer_number)

        self.use_actual_seqlen = config.use_eod_attn_mask_compression
        self.cp = 1 if self.config.context_parallel_size is None else self.config.context_parallel_size
        self.compute_2d = (config.sequence_parallel and self.cp == 1)

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        if config.multi_latent_attention:
            hidden_size_per_attention_head = config.qk_head_dim + config.qk_pos_emb_head_dim
        else:
            hidden_size_per_attention_head = projection_size // config.num_attention_heads

        # MindSpore FlashAttentionScore
        self.head_num = config.num_attention_heads
        self.input_layout = config.input_layout
        self.sparse_mode = config.sparse_mode
        self.attention_dropout = config.attention_dropout if attention_dropout is None else attention_dropout

        pre_tokens = 2147483647
        next_tokens = 0
        scale_value = 1. / math.sqrt(hidden_size_per_attention_head) if softmax_scale is None else softmax_scale
        self.flash_attention = FlashAttentionScore(head_num=self.head_num,
                                                   keep_prob=1 - self.attention_dropout,
                                                   scale_value=scale_value,
                                                   pre_tokens=pre_tokens,
                                                   next_tokens=next_tokens,
                                                   inner_precise=0,
                                                   input_layout=self.input_layout,
                                                   sparse_mode=self.sparse_mode)

        # Note: only support config.apply_query_key_layer_scaling=False,
        # FusedScaleMaskSoftmax does not require implementation.

        self.use_alibi_mask = config.use_alibi_mask
        self.use_mqa = config.num_query_groups == 1
        self.use_ring_attention = config.use_ring_attention
        self.use_attention_mask = not self.use_ring_attention
        self.enable_dropout = self.attention_dropout > 0.0

        if self.use_ring_attention:
            self.flash_attention.add_prim_attr("enable_ring_attention", True)
            self.flash_attention.add_prim_attr("enable_ra_send_recv", True)
        if self.use_alibi_mask:
            self.alibi_rescale_factor = Tensor([1.0 / scale_value], dtype=mstype.float16)
            self.alibi_rescale_mul = ops.Mul()
        if self.enable_dropout:
            self.keep_prob_tensor = Tensor(1 - self.attention_dropout, dtype=mstype.float16)
            self.drop_gen_mask = ops.DropoutGenMask()

        self.bnsd_transpose = aclnn_ops.Transpose()
        self.merge_head_transpose = aclnn_ops.Transpose()
        self.shape = aclnn_ops.Shape()
        self.reshape = aclnn_ops.Reshape()
        self.fa_out_transpose = aclnn_ops.Transpose()

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        else:
            self.shard(config)

    def _generate_flash_attention_strategy(self, dp, tp, cp, cp_ds=1):
        """get FA generate strategies"""
        # ulysses fa strategy
        cp_co = cp // cp_ds
        # from (dp, cp, tp) to (dp, cp_co, cp_ds * tp)
        cp = cp_co
        tp = cp_ds * tp

        if self.input_layout == "TND" and cp > 1:
            layout = Layout(device_matrix=(dp, cp, tp), alias_name=("dp", "cp", "tp"))
            fa_strategies = (layout(("dp", "cp"), "tp", "None"),
                             layout("dp", "tp", "None"),
                             layout("dp", "tp", "None"),
                             layout("None", "None"),
                             layout("dp"),
                             layout("dp"))
            return fa_strategies

        kv_head_split_num = 1 if self.use_mqa else tp
        if self.input_layout == "BSH":
            if self.use_ring_attention:
                fa_strategies = ((dp, cp, tp),
                                 (dp, cp, kv_head_split_num),
                                 (dp, cp, kv_head_split_num))
            else:
                fa_strategies = ((dp, cp, tp),
                                 (dp, 1, kv_head_split_num),
                                 (dp, 1, kv_head_split_num))
        elif self.input_layout == "BNSD":
            if self.use_ring_attention:
                fa_strategies = ((dp, tp, cp, 1),
                                 (dp, kv_head_split_num, cp, 1),
                                 (dp, kv_head_split_num, cp, 1))
            else:
                fa_strategies = ((dp, tp, cp, 1),
                                 (dp, kv_head_split_num, 1, 1),
                                 (dp, kv_head_split_num, 1, 1))
        elif self.input_layout == "TH":
            fa_strategies = ((dp, tp),
                             (dp, tp),
                             (dp, tp))
        elif self.input_layout == "TND":
            fa_strategies = ((dp, tp, 1),
                             (dp, tp, 1),
                             (dp, tp, 1))

        if self.use_alibi_mask:
            fa_strategies += ((dp, tp, cp, 1),)
        if self.enable_dropout:
            fa_strategies += ((dp, tp, cp, 1),)
        if self.use_attention_mask:
            if self.sparse_mode in [0, 1]:
                fa_strategies += ((dp, 1, cp, 1),)
            elif self.sparse_mode in [2, 3, 8]:
                fa_strategies += ((1, 1),)
            else:
                raise RuntimeError(f"sparse_mode: {self.sparse_mode} is not support currently")
        if self.input_layout in ["TH", "TND"] or self.use_actual_seqlen:
            fa_strategies += ((dp,), (dp,),)
        return fa_strategies

    def construct(self,
                  query: Tensor,
                  key: Tensor,
                  value: Tensor,
                  attention_mask: Tensor,
                  attn_mask_type: AttnMaskType = None,
                  attention_bias: Tensor = None,
                  packed_seq_params=None,
                  # additional
                  alibi_mask=None,
                  prefix=None,
                  padding_mask=None,
                  actual_seq_qlen=None,
                  actual_seq_kvlen=None):
        """Forward process of the AttentionMaskMF"""
        if attn_mask_type:
            raise NotImplementedError("For FlashAttention, 'attn_mask_type' is not supported for now.")
        if attention_bias:
            raise NotImplementedError("For FlashAttention, 'attention_bias' is not supported for now.")
        if packed_seq_params:
            raise NotImplementedError("For FlashAttention, 'packed_seq_params' is not supported for now.")
        if attention_mask is not None:
            attention_mask = cast(attention_mask, ms.uint8)

        if self.input_layout == "TND":
            _, _, _, output = self.flash_attention(query,
                                                   key,
                                                   value,
                                                   alibi_mask,
                                                   None,
                                                   padding_mask,
                                                   attention_mask,
                                                   prefix,
                                                   actual_seq_qlen,
                                                   actual_seq_kvlen)
        elif self.input_layout == "BNSD":
            query = self.bnsd_transpose(query, (1, 2, 0, 3))
            key = self.bnsd_transpose(key, (1, 2, 0, 3))
            value = self.bnsd_transpose(value, (1, 2, 0, 3))
            bsz, _, q_seq_len, _ = query.shape
            _, _, kv_seq_len, _ = key.shape
            if self.enable_dropout:
                drop_mask_bits = self.reshape(
                    self.drop_gen_mask((bsz, self.head_num, q_seq_len, kv_seq_len), self.keep_prob_tensor),
                    (bsz, self.head_num, q_seq_len, kv_seq_len // 8))
            else:
                drop_mask_bits = None
            if self.use_alibi_mask:
                alibi_mask = self.alibi_rescale_mul(alibi_mask, F.cast(self.alibi_rescale_factor, alibi_mask.dtype))
            _, _, _, output = self.flash_attention(query,
                                                   key,
                                                   value,
                                                   alibi_mask,
                                                   drop_mask_bits,
                                                   padding_mask,
                                                   attention_mask,
                                                   prefix)
            output = self._merge_heads(output)

        return output

    def _merge_heads(self, x):
        """
        Convert a 4D input tensor to a 3D output tensor.

        Inputs:
            x: input tensor

        Output:
            x_merge: the 3D output tensor
        """
        x = self.merge_head_transpose(x, (0, 2, 1, 3))  # dp,tp,cp,1 -> dp,cp,tp,1
        bs, seq_len, n_head, head_dim = self.shape(x)
        if self.compute_2d:
            new_shape = (bs * seq_len, n_head * head_dim)
        else:
            new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        x_merge = self.fa_out_transpose(x_merge, (1, 0, 2))
        return x_merge

    def shard(self, config: MLATransformerConfig):
        """sharding for flash attention"""
        dp = 1 if config is None else config.data_parallel_size
        tp = 1 if config is None else config.tensor_model_parallel_size
        cp = 1 if config is None else config.context_parallel_size

        self.bnsd_transpose.shard(((cp, dp, tp, 1),))
        self.merge_head_transpose.shard(((dp, tp, 1, 1),))
        self.fa_out_transpose.shard(((dp, cp, tp),))

        fa_strategies = self._generate_flash_attention_strategy(dp, tp, cp)
        self.flash_attention.shard(fa_strategies)

        if self.use_alibi_mask:
            self.alibi_rescale_mul.shard(((dp, tp, cp, 1), (1,)))

        return self
