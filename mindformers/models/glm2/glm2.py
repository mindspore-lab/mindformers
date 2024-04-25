# Copyright 2023 Huawei Technologies Co., Ltd
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
"""ChatGLM2 model."""
import mindspore.ops as ops
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import Tensor
from mindpet.delta.ptuning2 import PrefixEncoder

import numpy as np
from mindformers.mindformer_book import MindFormerBook
from mindformers.modules import VocabEmbedding, EmbeddingOpParallelConfig
from mindformers.modules.layers import Linear
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.core.loss import CrossEntropyLoss
from mindformers.pet.tuners.pet_adapter import PetAdapter
from mindformers.version_control import get_tril
from mindformers.models.modeling_utils import PreTrainedModel

from ..utils import cell_reuse
from .glm2_config import ChatGLM2Config
from .glm2_modules import precompute_rotary_emb_cache
from .glm2_transformer import ChatGLM2Transformer
from ...tools.logger import logger

__all__ = ['ChatGLM2ForConditionalGeneration', 'ChatGLM2Model', 'ChatGLM2WithPtuning2']


class GLM2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ChatGLM2Config
    base_model_prefix = "glm2"


class ChatGLM2Model(GLM2PreTrainedModel):
    r"""
    The backbone of ChatGLM2 network

    Args:
        config (GLMConfig): The config of network.
    """

    def __init__(self, config: ChatGLM2Config, **kwargs):
        super(ChatGLM2Model, self).__init__(config, **kwargs)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.seq_length = config.seq_length
        self.compute_dtype = config.compute_dtype
        self.use_past = config.use_past
        self.is_first_iteration = True

        # vocab embedding
        embed_parallel_config = EmbeddingOpParallelConfig()
        embed_parallel_config.data_parallel = config.parallel_config.data_parallel
        embed_parallel_config.model_parallel = config.parallel_config.model_parallel
        embed_parallel_config.vocab_emb_dp = False
        self.embedding = VocabEmbedding(vocab_size=config.vocab_size, embedding_size=config.hidden_size,
                                        parallel_config=embed_parallel_config, param_init_type=config.param_init_type)
        self.embedding.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        # rotary embedding
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.rotary_pos_emb = precompute_rotary_emb_cache(
            seq_len=self.seq_length,
            dim=rotary_dim // 2,
            rope_ratio=config.rope_ratio,
            dtype=config.compute_dtype
        )

        self.encoder = ChatGLM2Transformer(config)

        self.output_layer = Linear(config.hidden_size,
                                   config.vocab_size,
                                   has_bias=False,
                                   param_init_type=config.param_init_type,
                                   compute_dtype=config.compute_dtype)
        self.output_layer.shard(strategy_matmul=((config.parallel_config.data_parallel, 1),
                                                 (config.parallel_config.model_parallel, 1)))
        if config.parallel_config.pipeline_stage > 1:
            self.output_layer.pipeline_stage = config.parallel_config.pipeline_stage - 1
        self.output_layer.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.tril = get_tril()
        self.ones = P.Ones()
        self.less = P.Less()
        self.gather = P.Gather()
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.mul = P.Mul()
        self.tile = ops.Tile()
        low_triangle = np.tril(np.ones((1, self.seq_length, self.seq_length)))
        self.low_triangle = Tensor(low_triangle, mstype.int32)

    def get_masks(self, batch_size, padding_mask=None, input_position=None):
        """Get attention mask."""
        # [1, seq_length, seq_length] -> [batch_size, seq_length, seq_length]
        low_triangle = self.tile(self.low_triangle, (batch_size, 1, 1))
        if padding_mask is not None:
            low_triangle = self.mul(low_triangle, self.expand_dims(padding_mask, 1))
        if self.use_past and padding_mask is not None:
            low_triangle -= self.expand_dims(padding_mask, -1) - 1
        attention_mask = self.less(low_triangle, 0.5)
        if self.use_past and not self.is_first_iteration:
            # [bs, 1, seq_len] for incremental infer
            attention_mask = self.gather(attention_mask.view(-1, self.seq_length), input_position, 0)
        # [bs, 1, seq_len, seq_len] for normal, [bs, 1, 1, seq_len] for incremental infer
        attention_mask = self.reshape(attention_mask, (batch_size, 1, -1, self.seq_length))
        return attention_mask

    def construct(self, input_ids, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, batch_valid_length=None, full_attention_mask=None, prefix_key_values=None,
                  block_tables=None, slot_mapping=None):
        """ChatGLM2 model."""
        _ = position_ids
        batch_size, _ = input_ids.shape
        if input_embeds is None:
            input_embeds, _ = self.embedding(input_ids)  # (bs, seq_len, hs)

        if full_attention_mask is None:
            # (bs, 1, seq_len, seq_len)
            full_attention_mask = self.get_masks(batch_size, attention_mask, input_position)
            full_attention_mask = full_attention_mask.type(mstype.uint8)

        # Run encoder.
        hidden_states = self.encoder(
            input_embeds, full_attention_mask, self.rotary_pos_emb,
            batch_valid_length=batch_valid_length, prefix_key_values=prefix_key_values, block_tables=block_tables,
            slot_mapping=slot_mapping)
        return hidden_states


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ChatGLM2ForConditionalGeneration(GLM2PreTrainedModel):
    r"""
    Provide gpt training loss or logits through network.

    Args:
        config (GLMConfig): The config of ChatGLM2Model.

    Returns:
        Tensor, the loss or logits of the network.
    """
    _support_list = MindFormerBook.get_model_support_list()['glm2']
    _support_list.extend(MindFormerBook.get_model_support_list()['glm3'])
    _support_list.extend(MindFormerBook.get_model_support_list()['codegeex2'])

    @cell_reuse
    def __init__(self, config: ChatGLM2Config, **kwargs):
        super(ChatGLM2ForConditionalGeneration, self).__init__(config, **kwargs)
        self.transformer = ChatGLM2Model(config=config)
        self.cast = P.Cast()
        self.gather = P.Gather()
        self.loss = CrossEntropyLoss(parallel_config=config.parallel_config)
        self.gmask = config.gmask_token_id
        self.bos_token_id = config.bos_token_id
        self.use_past = config.use_past
        self.is_first_iteration = True
        self.not_equal = P.NotEqual()
        self.add = P.Add()
        self.load_checkpoint(config)
        self.vocab_size = config.padded_vocab_size
        self.set_model_predict_config()

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """prepare inputs for generation."""
        input_position = kwargs.get("current_index", None)
        if input_position is not None:
            input_position = Tensor(input_position, mstype.int32)
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs["origin_inputs"]
        return {
            "input_ids": Tensor(input_ids, mstype.int32),
            "input_position": input_position
        }

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get ChatGLM2 model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        bs = input_ids.shape[0]
        slot_mapping = Tensor(np.ones(shape=tuple([bs])), mstype.int32)
        return input_ids, None, None, None, None, None, None, None, None, None, slot_mapping

    def set_dynamic_inputs(self):
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_input_position = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_init_reset = Tensor([False], mstype.bool_)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(dynamic_input_ids, None, dynamic_input_position, None, None, None, dynamic_init_reset,
                        dynamic_batch_valid_length, None, dynamic_block_tables, dynamic_slot_mapping)
        logger.info("Set dynamic input for glm.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.transformer.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.transformer.encoder.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attention.infer_attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids=None, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, prefix_key_values=None,
                  block_tables=None, slot_mapping=None):
        """ChatGLM2 for conditional generation model."""
        # input_ids: (bs, seq_len)
        # position_ids: (bs, seq_len)
        # attention_mask: (bs, seq_len)
        bs, seq_len = input_ids.shape
        hidden_states = self.transformer(
            input_ids=input_ids,
            input_position=input_position,
            position_ids=position_ids,
            attention_mask=attention_mask,
            input_embeds=input_embeds,
            batch_valid_length=batch_valid_length,
            prefix_key_values=prefix_key_values,
            block_tables=block_tables,
            slot_mapping=slot_mapping
        )
        lm_logits = self.transformer.output_layer(hidden_states)
        if lm_logits.dtype == mstype.bfloat16:
            lm_logits = self.cast(lm_logits, mstype.float32)
        outputs = (lm_logits,)

        # train
        if labels is not None:
            logits = lm_logits.to(mstype.float32)
            labels = labels.reshape((-1,))
            logits = logits.reshape((-1, logits.shape[-1]))
            input_mask = self.not_equal(labels, -100).to(mstype.float32)
            input_mask = input_mask.reshape((-1,))

            if self.training:
                # if training, return loss directly
                outputs = self.loss(logits, labels, input_mask)
            else:
                # eval in train ppl
                # pre-shift to fit mindformers/core/metric/utils.py:PerplexityCell
                zeros = ops.zeros((bs, 1, self.vocab_size), dtype=logits.dtype)
                logits = logits.reshape((bs, seq_len, self.vocab_size))
                logits = ops.cat((logits, zeros), axis=1)

                zeros = ops.zeros((bs, 1), dtype=labels.dtype)
                labels = labels.reshape((bs, seq_len))
                labels = ops.cat((zeros, labels), axis=1)

                zeros = zeros.to(input_mask.dtype)
                input_mask = input_mask.reshape((bs, seq_len))
                input_mask = ops.cat((zeros, input_mask), axis=1)

                outputs = logits, labels, input_mask

        # generation process
        if (not self.use_past or self.is_first_iteration) and input_position is not None:
            lm_logits = lm_logits.reshape((-1, lm_logits.shape[-1]))
            lm_logits = self.gather(lm_logits, input_position, 0)
            outputs = (lm_logits,)

        return outputs


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ChatGLM2WithPtuning2(ChatGLM2ForConditionalGeneration):
    """
    ChatGLM2 Model for pretraining with p-tuning-v2

    Args:
        config (ChatGLM2Config): The config of network.
    """

    def __init__(self, config: ChatGLM2Config = None, **kwargs):
        ckpt_cfg = config.checkpoint_name_or_path
        config.checkpoint_name_or_path = None
        config.pre_seq_len = config.pet_config.pre_seq_len

        super().__init__(config, **kwargs)

        # get Pet tuning model.
        self.use_past = config.use_past
        config.pet_config.num_layers = config.num_layers
        config.pet_config.kv_channels = config.kv_channels
        if config.multi_query_attention:
            config.pet_config.num_heads = config.multi_query_group_num
        else:
            config.pet_config.num_heads = config.num_attention_heads
        self.prefix_encoder = PrefixEncoder(
            config.pet_config.pre_seq_len,
            config.pet_config.num_layers,
            config.pet_config.num_heads,
            config.pet_config.kv_channels,
            config.pet_config.prefix_projection,
            config.pet_config.projection_dim,
            config.pet_config.dropout_prob
        )

        if ckpt_cfg:
            # load ckpt
            config.checkpoint_name_or_path = ckpt_cfg
            self.load_checkpoint(config)

        # freeze pretrained model
        PetAdapter.freeze_pretrained_model(self, config.pet_config.pet_type)

    def construct(self, input_ids=None, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, prefix_key_values=None,
                  block_tables=None, slot_mapping=None):

        if not self.use_past or self.is_first_iteration:
            batch_size = input_ids.shape[0]
            prefix_key_values = self.prefix_encoder(batch_size)

        return super().construct(
            input_ids=input_ids,
            labels=labels,
            input_position=input_position,
            position_ids=position_ids,
            attention_mask=attention_mask,
            input_embeds=input_embeds,
            batch_valid_length=batch_valid_length,
            prefix_key_values=prefix_key_values
        )
