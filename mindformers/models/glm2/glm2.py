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
import mindspore as ms
import mindspore.ops as ops
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import Tensor, nn
from mindspore.common.initializer import initializer
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode
from mindpet.delta.ptuning2 import PrefixEncoder

import numpy as np
from mindformers.tools import logger
from mindformers.mindformer_book import MindFormerBook
from mindformers.modules.layers import Linear
from mindformers.modules import VocabEmbedding, EmbeddingOpParallelConfig
from mindformers.modules.transformer.transformer import LowerTriangularMaskWithDynamic
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.core.loss import CrossEntropyLoss
from mindformers.pet.tuners.pet_adapter import PetAdapter
from mindformers.version_control import get_tril, check_valid_paged_attention
from mindformers.modules import KVCachePreprocess

from ..base_model import BaseModel
from ..utils import cell_reuse
from .glm2_config import ChatGLM2Config
from .glm2_modules import RopeCache
from .glm2_transformer import ChatGLM2Transformer

__all__ = ['ChatGLM2ForConditionalGeneration', 'ChatGLM2WithPtuning2']


class ChatGLM2Model(nn.Cell):
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
        self.pre_seq_len = config.pre_seq_len

        self.is_dynamic = config.is_dynamic
        self.use_kvcache_op = config.use_kvcache_op
        self.is_flexible_shape = config.is_flexible_shape
        self.use_paged_attention = config.use_paged_attention and check_valid_paged_attention()
        if self.use_paged_attention:
            logger.info("Enable paged attention.")

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)

        # rotary embedding
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.rope_cache = RopeCache(config=config, dim=rotary_dim // 2, is_dynamic=config.is_dynamic)
        use_flash_attention_flag = config.use_flash_attention or config.use_prompt_flash_attention
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          mask_type=config.mask_type,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=use_flash_attention_flag)

        max_seq_length = config.seq_length if not self.pre_seq_len else config.seq_length + self.pre_seq_len
        self.kvcache_preprocess = KVCachePreprocess(max_batch_size=config.batch_size,
                                                    max_seq_length=max_seq_length,
                                                    is_dynamic=config.is_dynamic,
                                                    use_kvcache_op=config.use_kvcache_op,
                                                    is_flexible_shape=config.is_flexible_shape,
                                                    use_paged_attention=self.use_paged_attention)

        # vocab embedding
        embed_parallel_config = EmbeddingOpParallelConfig()
        embed_parallel_config.data_parallel = config.parallel_config.data_parallel
        embed_parallel_config.model_parallel = config.parallel_config.model_parallel
        embed_parallel_config.vocab_emb_dp = config.parallel_config.vocab_emb_dp
        self.embedding = VocabEmbedding(vocab_size=config.vocab_size, embedding_size=config.hidden_size,
                                        parallel_config=embed_parallel_config,
                                        param_init=initializer('normal',
                                                               [config.vocab_size, config.hidden_size],
                                                               dtype=config.embedding_type),
                                        )
        self.embedding.embedding_table.parallel_optimizer = True
        if config.parallel_config.pipeline_stage > 1:
            self.embedding.pipeline_stage = 0
            self.embedding.set_comm_fusion(2)
        else:
            self.embedding.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.encoder = ChatGLM2Transformer(config)

        self.output_layer = Linear(config.hidden_size,
                                   config.vocab_size,
                                   has_bias=False,
                                   param_init_type=config.param_init_type,
                                   compute_dtype=config.compute_dtype,
                                   skip_redistribution=config.is_dynamic)

        self.output_layer.shard(strategy_matmul=((config.parallel_config.data_parallel, 1),
                                                 (config.parallel_config.model_parallel, 1)))

        if config.parallel_config.pipeline_stage > 1:
            self.output_layer.pipeline_stage = config.parallel_config.pipeline_stage - 1
            self.output_layer.set_comm_fusion(2)
        else:
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

    def construct(self, input_ids, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_key_values=None):
        """ChatGLM2 model."""
        batch_size, seq_len = self.shape(input_ids)
        input_embeds, _ = self.embedding(input_ids)

        if not self.use_past:
            rotary_pos_emb = self.rope_cache()
            full_attention_mask = self.casual_mask(input_ids)  # full_attention_mask: [bs, seq, seq]
            full_attention_mask = self.casual_mask.post_process(full_attention_mask)
            kvcache_inputs = None
        else:
            if self.is_first_iteration:
                rotary_pos_emb = self.rope_cache(seq_len)  # 2048, 32, 2
                full_attention_mask = self.casual_mask(input_ids)  # full_attention_mask: [bs, seq, seq]
            else:
                rotary_pos_emb = self.rope_cache.increment(batch_valid_length, batch_size)
                if self.is_dynamic and self.is_flexible_shape and not self.use_kvcache_op:
                    full_attention_mask = self.casual_mask.increment_slice(
                        self.kvcache_preprocess.range,
                        self.kvcache_preprocess.max_cache_length // batch_size, batch_valid_length,
                        zactivate_len)
                else:
                    full_attention_mask = self.casual_mask.increment(self.kvcache_preprocess.range,
                                                                     batch_valid_length, zactivate_len)

            full_attention_mask = self.casual_mask.post_process(full_attention_mask)

            if batch_valid_length is not None and isinstance(self.pre_seq_len, int):
                batch_valid_length = batch_valid_length + self.pre_seq_len

            kvcache_inputs = self.kvcache_preprocess(batch_size, batch_valid_length, batch_index, zactivate_len,
                                                     block_tables, slot_mapping)

        # Run encoder.
        hidden_states = self.encoder(
            input_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb, kvcache_inputs=kvcache_inputs,
            prefix_key_values=prefix_key_values)

        return hidden_states


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ChatGLM2ForConditionalGeneration(BaseModel):
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
        self.config = config
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.sub_batch_valid_len = P.Sub()

        self.max_seq_len = config.max_length
        self.transformer = ChatGLM2Model(config=config)
        self.cast = P.Cast()
        self.gather = P.Gather()
        self.is_first_iteration = True
        self.loss = CrossEntropyLoss(parallel_config=config.parallel_config)
        self.gmask = config.gmask_token_id
        self.bos_token_id = config.bos_token_id
        self.use_past = config.use_past
        self.is_first_iteration = True
        self.not_equal = P.NotEqual()
        self.add = P.Add()
        self.load_checkpoint(config)
        self.vocab_size = config.padded_vocab_size
        dp = config.parallel_config.data_parallel
        if _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            self.gather.shard(((dp, 1), (dp,)))

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """prepare inputs for generation."""
        input_position = kwargs.get("current_index", None)
        if input_position is not None:
            input_position = Tensor(input_position, mstype.int32)
        return {
            "input_ids": Tensor(input_ids, mstype.int32),
            "input_position": input_position
        }

    def prepare_inputs_for_export(self, full_model=True):
        dyn = self.config.is_dynamic
        use_paged_attention = self.config.use_paged_attention and check_valid_paged_attention()
        use_past = self.config.use_past
        if dyn:
            logger.info(f"Exporting dynamic MindIR...")
        if use_paged_attention:
            logger.info(f"Exporting model with paged attention...")
        seq_length = self.config.seq_length
        bs = None if dyn else self.config.batch_size
        seq_len = None if dyn else self.config.seq_length

        max_num_blocks_pre_batch = None if dyn else seq_len // self.config.block_size
        logger.info(f"max num blocks pre batch: {max_num_blocks_pre_batch}")

        def dummy_tensor(shape, dtype):
            if None in shape:
                return Tensor(shape=shape, dtype=dtype)
            return Tensor(np.ones(shape=tuple(shape)), dtype=dtype)

        batch_valid_length = dummy_tensor(shape=[bs], dtype=ms.int32) if use_past else None
        batch_index = None if use_paged_attention else dummy_tensor(shape=[bs], dtype=ms.int64)
        zactivate_len = None if use_paged_attention else dummy_tensor(shape=[seq_len], dtype=ms.int64)
        prefill_mapping_len = None if dyn else bs * seq_len
        inc_mapping_len = None if dyn else bs * 1

        if full_model:
            logger.info('\nexporting with batch_size = %s, seq = %s ...', self.config.batch_size, seq_length)
            input_ids = dummy_tensor(shape=[bs, seq_len], dtype=ms.int32)
            slot_mapping = dummy_tensor(shape=[prefill_mapping_len], dtype=ms.int32) if use_paged_attention else None
            block_tables = None
        else:
            logger.info('\nexporting with batch_size = %s, seq = 1 ...', self.config.batch_size)
            input_ids = dummy_tensor(shape=[bs, 1], dtype=ms.int32)
            slot_mapping = dummy_tensor(shape=[inc_mapping_len], dtype=ms.int32) if use_paged_attention else None
            block_tables = dummy_tensor(shape=[inc_mapping_len, max_num_blocks_pre_batch],
                                        dtype=ms.int32) if use_paged_attention else None
        return input_ids, None, None, None, None, None, None, batch_valid_length, None, batch_index, zactivate_len, \
               block_tables, slot_mapping

    # pylint: disable=W0613
    def construct(self, input_ids=None, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, prefix_key_values=None,
                  batch_index=None, zactivate_len=None, block_tables=None, slot_mapping=None):
        """ChatGLM2k for conditional generation model."""
        bsz, seqlen = self.shape(input_ids)
        tokens = input_ids
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        if not self.is_first_iteration:
            batch_valid_length = self.sub_batch_valid_len(batch_valid_length, 1)

        hidden_states = self.transformer(
            input_ids=tokens,
            batch_valid_length=batch_valid_length,
            batch_index=batch_index,
            zactivate_len=zactivate_len,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            prefix_key_values=prefix_key_values
        )
        # gather前移
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            hidden_states = self.reshape(hidden_states, (-1, hidden_states.shape[-1]))
            hidden_states = self.gather(hidden_states, self.sub_batch_valid_len(batch_valid_length, 1), 0)

        lm_logits = self.transformer.output_layer(hidden_states)
        if not self.training and not pre_gather:
            lm_logits = self.reshape(lm_logits, (bsz, seqlen, -1))

        outputs = (lm_logits,)

        # train
        if labels is not None:
            logits = self.cast(lm_logits, mstype.float32)
            logits_shape = logits.shape
            labels = self.reshape(labels, (-1,))
            logits = self.reshape(logits, (-1, logits_shape[-1]))
            input_mask = self.not_equal(labels, -100).to(mstype.float32)
            input_mask = self.reshape(input_mask, (-1,))

            if self.training:
                # if training, return loss directly
                outputs = self.loss(logits, labels, input_mask)
            else:
                # eval in train ppl
                # pre-shift to fit mindformers/core/metric/utils.py:PerplexityCell
                zeros = ops.zeros((bsz, 1, self.vocab_size), dtype=logits.dtype)
                logits = logits.reshape((bsz, seqlen, self.vocab_size))
                logits = ops.cat((logits, zeros), axis=1)

                zeros = ops.zeros((bsz, 1), dtype=labels.dtype)
                labels = labels.reshape((bsz, seqlen))
                labels = ops.cat((zeros, labels), axis=1)

                zeros = zeros.to(input_mask.dtype)
                input_mask = input_mask.reshape((bsz, seqlen))
                input_mask = ops.cat((zeros, input_mask), axis=1)

                outputs = logits, labels, input_mask

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
                  batch_index=None, zactivate_len=None, block_tables=None, slot_mapping=None):

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
            init_reset=init_reset,
            batch_valid_length=batch_valid_length,
            prefix_key_values=prefix_key_values,
            batch_index=batch_index,
            zactivate_len=zactivate_len,
            block_tables=block_tables,
            slot_mapping=slot_mapping
        )
