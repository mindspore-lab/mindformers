# coding=utf-8
# Copyright 2022 The OpenBMB team.
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

import os
from typing import Optional, List, Tuple
import mindspore as ms
from mindspore import nn, ops
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Normal
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from .layers import TransformerBlock, BucketPositionBias, EmbeddingExt, LayerNorm, masked_fill
from .cpm_config import CPMBeeConfig


#  Get MS backend: 0 vm 1 GE
is_ge = os.getenv('MS_ENABLE_GE')
if is_ge == '1':
    jit_level = "O3"
else:
    jit_level = "O1"

__all__ = ['CPMForPreTraining']


class Encoder(nn.Cell):
    """Layers of encoder transformer blocks plus an final layernorm.

    Args:
        num_layers (int): number of layers.
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-6.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        dim_ff: int,
        num_heads: int,
        dim_head: int,
        dtype: mstype.float_ = mstype.half,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
    ):

        super().__init__()

        self.num_layers = num_layers

        if mask_modules is not None:
            assert (
                len(mask_modules) == num_layers
            ), "The total number of masks should equal to num_layers"
            for mask_module in mask_modules:
                assert (
                    len(mask_module) == 2
                ), "For encoder, each mask should be (mask_att, mask_ffn)"
        else:
            mask_modules = [(False, False)] * num_layers

        self.layers = nn.CellList()

        for ith in range(num_layers):
            block = TransformerBlock(
                        dim_model=dim_model,
                        dim_ff=dim_ff,
                        num_heads=num_heads,
                        dim_head=dim_head,
                        dtype=dtype,
                        eps=eps,
                        dropout_p=dropout_p,
                        mask_att=mask_modules[ith][0],
                        mask_ffn=mask_modules[ith][1],
                    )
            block.recompute()
            self.layers.append(block)

        self.output_layernorm = LayerNorm(dim_norm=dim_model, dtype=dtype, eps=eps)

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_bias: Tensor,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
    ):
        """
        Args:
            hidden-states (:obj:`Tensor` of shape ``(batch, seq_enc, dim_model)``): Input of encoder, might be the embedding of a batch of sequences.
            attention_mask (:obj:`Tensor` of shape ``(batch, seq_enc, seq_enc)``): Avoid invalid areas to participate in the calculation
            position_bias(:obj:`Tensor` of shape ``(num_heads, seq_enc, seq_enc)``) Provides position information to attention mechanism.

        Return:
            :obj:`Tensor` of shape ``(batch, seq_enc, dim_model)``: The encoder output.

        """  # noqa: E501
        if not use_cache:
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask, position_bias)
            hidden_states = self.output_layernorm(hidden_states)
            return hidden_states
        else:
            current_key_values = []
            for i, module in enumerate(self.layers):
                hidden_states = module(
                    hidden_states,
                    attention_mask,
                    position_bias,
                    past_key_value=past_key_values[i] if past_key_values else None,
                    use_cache=use_cache,
                )
                if use_cache:
                    current_key_values.append(hidden_states[1])
                    hidden_states = hidden_states[0]
            hidden_states = self.output_layernorm(hidden_states)
            if use_cache:
                return hidden_states, current_key_values
            else:
                return hidden_states

    def shard(self, dp, mp):
        for cell in self.layers:
            cell.shard(dp, mp)
        self.output_layernorm.shard(dp, mp)


class CPMBee(nn.Cell):
    def __init__(self, config: CPMBeeConfig):
        super().__init__()

        self.encoder = Encoder(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            dim_ff=config.dim_ff,
            num_heads=config.num_heads,
            dim_head=config.dim_head,
            dtype=config.dtype,
            eps=config.eps,
            dropout_p=config.dropout_p,
            mask_modules=config.mask_modules,
        )

        self.input_embedding = EmbeddingExt(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
            dtype=config.dtype,
            param_init=Normal(sigma=0.02),
        )

        self.position_bias = BucketPositionBias(
            num_heads=config.num_heads,
            num_buckets=config.position_bias_num_buckets,
            num_segment_bucket=config.position_bias_num_segment_buckets,
            max_distance=config.position_bias_max_distance,
            dtype=config.dtype,
        )

        self.seg_log_and = ops.LogicalAnd()
        self.seg_rel_log_and = ops.LogicalAnd()
        self.sample_mask_or = ops.BitwiseOr()

        self.att_logit_or = ops.LogicalOr()
        self.att_logit_and = ops.LogicalAnd()
        self.att_mask_and = ops.LogicalAnd()
        self.att_mask_and2 = ops.LogicalAnd()
        self.log_and = ops.LogicalAnd()
        self.att_1d_mask_and = ops.LogicalAnd()

    def construct(
            self,
            input: Tensor,  # (batch, seqlen) int32
            input_sub: Tensor,  # (batch, seqlen) int32
            length: Tensor,  # (batch) int32
            context: Tensor,  # (batch, seqlen) bool
            sample_ids: Tensor,  # (batch, seq_len) int32
            num_segments: Tensor,  # (batch, seq_len) int32
            segment: Tensor,  # (batch, seqlen) int32
            segment_rel_offset: Tensor,  # (batch, seq_len) int32
            segment_rel: Tensor,  # (batch, num_segment_bucket) int32
            span: Tensor,  # (batch, seqlen) int32
            ext_table_ids: Tensor,  # (ext_table_size) int32
            ext_table_sub: Tensor,  # (ext_table_size) int32
    ):
        batch = input.shape[0]
        seqlen = input.shape[1]
        # processing masks and position bias bucket

        context = ops.cast(context, mstype.bool_)
        ext_table_sub = ext_table_sub[0]
        ext_table_ids = ext_table_ids[0]
        # calc segment bucket
        segment_rel_2d = ops.masked_fill(
            segment[:, :, None] * num_segments[:, :, None]
            + segment[:, None, :]
            + segment_rel_offset[:, :, None],
            ~(
                self.seg_rel_log_and(
                    (sample_ids[:, :, None] == sample_ids[:, None, :]),
                    (span[:, None, :] == span[:, :, None]))
            ),  # not in the same span or sample
            0,  # avoid torch.gather overflow
        ).view((batch, seqlen * seqlen))

        segment_bucket = ops.gather_elements(
            input=segment_rel,
            dim=1,
            index=segment_rel_2d.astype(mstype.int32),
        ).view((batch, seqlen, seqlen))

        segment_bucket = masked_fill(
            segment_bucket,
            ~(
                self.seg_log_and((sample_ids[:, :, None] == sample_ids[:, None, :]),
                                 (span[:, None, :] == span[:, :, None]))
            ),  # not in the same span or sample
            1,  # bucket is used for in-context samples
        )

        # directional mask
        directional_mask_2d = ops.arange(seqlen) <= ops.arange(seqlen).view((-1, 1))
        # sample mask
        sample_mask_2d = self.sample_mask_or((sample_ids[:, :, None] == 0).to(mstype.int32),
                                             (sample_ids[:, :, None] == sample_ids[:, None, :]).to(mstype.int32)
                                             ).to(mstype.bool_)

        # context mask
        attention_mask = self.att_logit_or(context[:, None, :],
                                           (self.att_logit_and(ops.logical_not(context[:, :, None]),
                                                               directional_mask_2d.view((1, seqlen, seqlen)))
                                            ))
        # span mask
        attention_mask = (
                attention_mask.to(mstype.int32) & sample_mask_2d.to(mstype.int32) & \
                (span[:, None, :] == span[:, :, None]).to(mstype.int32)
        )
        # length mask
        mask_1d = (
                ops.arange(seqlen, dtype=mstype.int32)[None, :].tile((batch, 1)) < length[:, None]
        )

        mask_1d_and = self.log_and(mask_1d.view((batch, seqlen, 1)), mask_1d.view((batch, 1, seqlen)))
        attention_mask = mask_1d_and.to(mstype.int32) & attention_mask
        position = ops.arange(seqlen, dtype=mstype.int32).broadcast_to((batch, seqlen))

        position = ops.stop_gradient(position)
        segment_bucket = ops.stop_gradient(segment_bucket)
        attention_mask = ops.stop_gradient(attention_mask)

        hidden_states = self.input_embedding(input, input_sub)
        position_bias = self.position_bias(position, position, segment_bucket)
        hidden_states = self.encoder(hidden_states, attention_mask, position_bias)

        ext_table = self.input_embedding(ext_table_ids, ext_table_sub)

        logits = self.input_embedding.projection(hidden_states, ext_table)

        return logits, hidden_states

    @staticmethod
    def prepare_data(
            input,
            input_sub,
            length,
            context,
            sample_ids,
            num_segments,
            segment,
            segment_rel_offset,
            segment_rel,
            span,
            ext_table_ids,
            ext_table_sub,
            label
    ):
        return_list = [input, input_sub, length, context, sample_ids, num_segments, segment, segment_rel_offset,
                       segment_rel, span, ext_table_ids, ext_table_sub, label]
        return tuple(Tensor(i) for i in return_list)

    def shard(self, dp, mp):
        self.input_embedding.shard(dp, mp)
        self.encoder.shard(dp, mp)
        self.position_bias.shard(dp, mp)


class DefineSoftmaxCrossEntropyWithLogits(nn.Cell):
    def __init__(self, sparse=False):
        super(DefineSoftmaxCrossEntropyWithLogits, self).__init__()
        self.sparse = sparse
        self.loss_fun = nn.SoftmaxCrossEntropyWithLogits(sparse=self.sparse)

    def construct(self, inputs, target, ignore_index=-100):
        loss = self.loss_fun(inputs, target)
        new_target = ops.where(target != ignore_index, ms.Tensor(1, dtype=ms.float32), ms.Tensor(0, dtype=ms.float32))
        loss = ops.sum(loss * new_target) / ops.sum(new_target)
        return loss


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class CPMForPreTraining(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.model = CPMBee(config)
        # self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(True, 'none')
        self.loss_fn = DefineSoftmaxCrossEntropyWithLogits(sparse=True)
        self.cast = ops.Cast()
        self.shard(int(os.getenv('RANK_SIZE')), 1)

    def construct(
            self,
            input: Tensor,  # (batch, seqlen) int32
            input_sub: Tensor,  # (batch, seqlen) int32
            length: Tensor,  # (batch) int32
            context: Tensor,  # (batch, seqlen) bool
            sample_ids: Tensor,  # (batch, seq_len) int32
            num_segments: Tensor,  # (batch, seq_len) int32
            segment: Tensor,  # (batch, seqlen) int32
            segment_rel_offset: Tensor,  # (batch, seq_len) int32
            segment_rel: Tensor,  # (batch, num_segment_bucket) int32
            span: Tensor,  # (batch, seqlen) int32
            ext_table_ids: Tensor,  # (ext_table_size) int32
            ext_table_sub: Tensor,  # (ext_table_size) int32
            label: Tensor
    ):
        logits, _ = self.model(input, input_sub, length, context, sample_ids,
                               num_segments, segment, segment_rel_offset, segment_rel, span,
                               ext_table_ids, ext_table_sub)
        logits = logits.astype(mstype.float32)
        loss = self.loss_fn(logits.view((-1, logits.shape[-1])), label.view(-1))
        return loss

    def shard(self, dp, mp):
        self.model.shard(dp, mp)
        self.loss_fn.loss_fun.softmax_cross_entropy.shard(((dp, 1), (dp, 1)))
        self.loss_fn.loss_fun.sparse_softmax_cross_entropy.shard(((dp, 1), (dp, 1)))
