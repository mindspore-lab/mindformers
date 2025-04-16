# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
import copy
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Condition

import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import Tensor, mint, Parameter
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindpet.delta.ptuning2 import PrefixEncoder

import numpy as np
from safetensors import safe_open

from mindformers.mindformer_book import MindFormerBook
from mindformers.modules.transformer.transformer import LowerTriangularMaskWithDynamic
from mindformers.modules.layers import Linear
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_predict_run_mode
from mindformers.core.loss import CrossEntropyLoss
from mindformers.pet.tuners.pet_adapter import PetAdapter
from mindformers.version_control import get_dropout
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.glm2.glm2_modules import GLMEmbedding
from mindformers.version_control import check_safetensors_addition_param_support

from ..utils import lazy_inline
from .glm2_config import ChatGLM2Config
from .glm2_modules import FreqsMgr, FreqsMgrRope, GetEodResetMask
from .glm2_transformer import ChatGLM2Transformer
from ...tools.logger import logger

__all__ = ['ChatGLM2ForConditionalGeneration', 'ChatGLM2Model', 'ChatGLM2WithPtuning2']

# For long sequence, activation checkpointing of AllGather in case of tp-sp is too large, and should be eliminated;
# For short sequence, static memory especially gradient accumulation is too large, and should be sharded by tp.
SHARDING_SEQ_LEN_THRESHOLD = 32768


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
        self.seq_length = config.seq_length
        self.compute_dtype = config.compute_dtype
        self.use_past = config.use_past
        self.use_flash_attention = config.use_flash_attention
        self.is_first_iteration = True
        self.use_rearrange_rope = config.use_rearrange_rope
        self.use_ring_attention = config.use_ring_attention
        self.n_kv_head = config.num_heads if config.n_kv_heads is None else config.n_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.seq_split_num = config.parallel_config.seq_split_num
        self.seq_pipe = self.seq_split_num > 1
        # vocab embedding
        dp, cp, mp = _parallel_decompose(config)
        kv_mp = self.n_kv_head if self.n_kv_head < mp else mp

        if self.seq_pipe:
            if self.use_ring_attention:
                raise ValueError(f"When the seq_pipe = True, the use_ring_attention cannot be True ")
            kv_shape = (config.batch_size * dp, config.seq_length, self.n_kv_head * self.head_dim)
            self.zeros = initializer('zeros', kv_shape, dtype=self.compute_dtype)
            self.seq_update = Tensor(1, dtype=mstype.int32)
            self.seq_zero = Tensor(0, dtype=mstype.int32)
            self.seq_seg_len = config.seq_length // self.seq_split_num
            kv_mask = np.zeros((1, config.seq_length, self.n_kv_head * self.head_dim), np.int32)
            for s in range(self.seq_split_num):
                kv_mask[:, s * self.seq_seg_len: (s + 1) * self.seq_seg_len, :] = s
            self.kv_mask = Tensor(kv_mask)
            self.seq_chunk = Parameter(Tensor(0, dtype=mstype.int32), name="seq_chunk",
                                       requires_grad=False, parallel_optimizer=False)
            self.equal_kv = P.Equal().shard(((dp, cp, kv_mp), ()))
            self.kv_mask_add = P.Add().shard(((dp, cp, kv_mp), (1, cp, kv_mp)))
            self.assign_add_count = P.AssignAdd()
            self.assign_count = P.Assign()
            self.assign_mask = P.Assign().shard(((dp, 1), (dp, 1)))
            self.mask_zeros = Tensor(np.zeros((config.batch_size * dp, config.seq_length)), mstype.float32)

        # mask
        total_batch_size_in_dp = config.batch_size * config.parallel_config.data_parallel
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          batch_size=total_batch_size_in_dp,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention,
                                                          use_past=config.use_past,
                                                          seq_split_num=self.seq_split_num)

        # vocab embedding
        self.embedding = GLMEmbedding(vocab_table_size=config.vocab_size, embedding_size=config.hidden_size,
                                      param_init_type=config.param_init_type, parallel_optimizer=True)
        # rotary embedding
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        freq_mgr_class = FreqsMgrRope if self.use_rearrange_rope else FreqsMgr
        self.freqs_mgr = freq_mgr_class(dim=rotary_dim // 2,
                                        seq_length=config.seq_length,
                                        rotary_dtype=config.rotary_dtype,
                                        base=10000,
                                        rope_ratio=config.rope_ratio,
                                        parallel_config=config.parallel_config)

        self.encoder = ChatGLM2Transformer(config)
        self.output_layer = Linear(config.hidden_size,
                                   config.vocab_size,
                                   has_bias=False,
                                   param_init_type=config.param_init_type,
                                   compute_dtype=config.compute_dtype)
        self.shard_head_tail(config)

        # mask
        self.less = P.Less()
        self.gather = P.Gather()
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.mul = P.Mul()
        self.tile = ops.Tile()
        self.low_triangle = Tensor(np.tril(np.ones((1, self.seq_length, self.seq_length))), mstype.int32)
        self.cast = P.Cast()
        self.dropout = get_dropout(config.hidden_dropout)
        self.dropout.dropout.shard(((dp, cp, 1),))
        if config.parallel_config.use_seq_parallel:
            self.dropout.dropout.shard(((dp, cp * mp, 1),))
        parallel_config = config.parallel_config
        self.mask_generate = config.mask_generate  # "inmap", "compress_reset"
        self.get_attention_mask = GetEodResetMask(seq_length=config.seq_length, parallel_config=parallel_config)

    def shard_head_tail(self, config):
        """shard embedding head and lm head"""
        dp, cp, mp = _parallel_decompose(config)
        use_sp = config.parallel_config.use_seq_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.embedding.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.embedding.set_comm_fusion(2)
                self.output_layer.pipeline_stage = config.parallel_config.pipeline_stage - 1
            else:
                self.embedding.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
            if config.parallel_config.vocab_emb_dp or (config.vocab_size % (dp * mp * cp) != 0):
                if use_sp:
                    if self.seq_length >= SHARDING_SEQ_LEN_THRESHOLD:
                        self.output_layer.shard(strategy_matmul=((dp * cp * mp, 1), (1, 1)))
                    else:
                        self.output_layer.shard(strategy_matmul=((dp * cp, 1), (mp, 1)))
                else:
                    self.output_layer.shard(strategy_matmul=((dp * cp, 1), (1, 1)))
            else:
                self.output_layer.shard(strategy_matmul=((1, 1), (dp * cp * mp, 1)))

    def get_masks(self, batch_size, seq_len, padding_mask=None, input_position=None):
        """Get attention mask."""
        # [1, seq_len, seq_len] -> [batch_size, seq_len, seq_len]
        if seq_len < self.low_triangle.shape[-1]:
            low_triangle = self.low_triangle[..., :seq_len, :seq_len]
        else:
            low_triangle = self.low_triangle
        low_triangle = self.tile(low_triangle, (batch_size, 1, 1))
        if padding_mask is not None:
            low_triangle = self.mul(low_triangle, self.expand_dims(padding_mask, 1))
        if self.use_past and padding_mask is not None:
            low_triangle -= self.expand_dims(padding_mask, -1) - 1
        attention_mask = self.less(low_triangle, 0.5)
        if self.use_past and not self.is_first_iteration:
            # [bs, 1, seq_len] for incremental infer
            attention_mask = self.gather(attention_mask.view(-1, seq_len), input_position, 0)
        # [bs, 1, seq_len, seq_len] for normal, [bs, 1, 1, seq_len] for incremental infer
        attention_mask = self.reshape(attention_mask, (batch_size, 1, -1, seq_len))
        return attention_mask

    # pylint: disable=W0613
    def construct(self, input_ids, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, batch_valid_length=None, full_attention_mask=None, prefix_key_values=None,
                  block_tables=None, slot_mapping=None):
        """ChatGLM2 model."""
        _ = position_ids
        batch_size, seq_len = input_ids.shape
        kv_mask = None
        mask = None
        seq_chunk = None
        if self.use_past:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr.prefill()
                mask = self.casual_mask.prefill()
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)
        else:
            if full_attention_mask is None:
                if attention_mask is None:
                    if self.seq_pipe:
                        mask = self.casual_mask(input_ids, seq_chunk=self.seq_chunk)
                        seq_chunk = P.ReLU()(self.seq_chunk)
                        kv_mask = self.cast(self.equal_kv(self.kv_mask_add(self.zeros, self.kv_mask), seq_chunk),
                                            self.compute_dtype)
                        seq_update = F.depend(self.seq_update, mask)
                        seq_update = F.depend(seq_update, kv_mask)
                        full_attention_mask = F.depend(mask, self.assign_add_count(self.seq_chunk, seq_update))
                    else:
                        # (bs, 1, seq_len, seq_len)
                        full_attention_mask = self.get_masks(batch_size, seq_len, attention_mask, input_position)
                        full_attention_mask = full_attention_mask.type(mstype.uint8)
                else:
                    if self.mask_generate == "inmap":
                        full_attention_mask = self.get_attention_mask(attention_mask)
                        full_attention_mask = F.reshape(full_attention_mask, (batch_size, 1, seq_len, seq_len))
                    else:
                        full_attention_mask = attention_mask
            mask = full_attention_mask
            freqs_cis = self.freqs_mgr(seq_len, seq_chunk=seq_chunk)
        if input_embeds is None:
            input_embeds = self.embedding(input_ids)  # (bs, seq_len, hs)
        input_embeds = self.dropout(input_embeds)

        # Run encoder.
        hidden_states = self.encoder(
            input_embeds, mask, freqs_cis,
            batch_valid_length=batch_valid_length, prefix_key_values=prefix_key_values, block_tables=block_tables,
            slot_mapping=slot_mapping, kv_mask=kv_mask, seq_chunk=seq_chunk)
        return hidden_states

    def clear_kv_cache(self):
        zeros = 0.0
        return_tuple = ()
        return_tuple += (self.assign_count(self.seq_chunk, self.seq_zero),)
        return_tuple += (self.assign_mask(self.casual_mask.mask_cache, self.mask_zeros),)
        return F.depend(zeros, return_tuple)


def _parallel_decompose(config):
    dp, cp, mp = config.parallel_config.data_parallel, \
                 config.parallel_config.context_parallel, config.parallel_config.model_parallel
    return dp, cp, mp


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ChatGLM2ForConditionalGeneration(GLM2PreTrainedModel):
    """
    Provide ChatGLM2 training loss or logits through network.

    Args:
        config (ChatGLM2Config): The config of ChatGLM2Model.
        kwargs (dict, optional): A variable number of keyword parameters reserved for the keyword parameters to be
            expanded.

    Inputs:
        - **input_ids** (Tensor, optional) - A tokenized input tensor, which is of int32 integer type and has a shape of
          (batch, seq_length). Default: ``None``.
        - **labels** (Tensor, optional) - A tokenized label tensor, which is of int32 integer type and has a shape of
          (batch, seq_length). Default: ``None``.
        - **input_position** (Tensor, optional) - The current position, used in predict. Default: ``None``.
        - **position_ids** (Tensor, optional) - Keep the parameter unused. Default: ``None``.
        - **attention_mask** (Tensor, optional) - Keep the parameter unused. Default: ``None``.
        - **input_embeds** (Tensor, optional) - Keep the parameter unused. Default: ``None``.
        - **init_reset** (Tensor, optional) - A bool tensor with shape [1], used to clear previous key-value pairs in
          incremental inference. Default: ``None``.
        - **batch_valid_length** (Tensor, optional) - In incremental inference, a tensor used for calculating the index
          of the previous step. It is of int32 type and has a shape of [batch_size]. Default: ``None``.
        - **prefix_key_values** (Tensor, optional) - A set of additional key-value pairs added before the regular
          key-value pairs. These prefix key-value pairs can be used to capture long-term dependencies or provide prior
          knowledge, thereby helping the model better understand and generate sequences. Default: ``None``.
        - **block_tables** (Tensor, optional) - Store the mapping table for each sequence. Default: ``None``.
        - **slot_mapping** (Tensor, optional) - Store the physical slot index of the sequence cache. Default: ``None``.
        - **batch_index** (Tensor, optional) - Keep the parameter unused. Default: ``None``.
        - **zactivate_len** (Tensor, optional) - Keep the parameter unused. Default: ``None``.
        - **input_mask** (Tensor, optional) - Mask for the input parts in input_ids. Default: ``None``.

    Outputs:
        output(Tensor), including an on-line loss value or a logical value, a sequence of predictive text, an input
        mask.

    Examples:
        >>> from mindformers.models.glm2 import ChatGLM2Config, ChatGLM2ForConditionalGeneration
        >>> config = ChatGLM2Config(batch_size=2)
        >>> network = ChatGLM2ForConditionalGeneration(config=config)
        >>> type(network)
        <class 'mindformers.models.glm2.glm2.ChatGLM2ForConditionalGeneration'>
        >>> from mindformers import ChatGLM2ForConditionalGeneration
        >>> network = ChatGLM2ForConditionalGeneration.from_pretrained('glm3_6b')
        >>> type(network)
        <class 'mindformers.models.glm2.glm2.ChatGLM2ForConditionalGeneration'>
    """
    _support_list = MindFormerBook.get_model_support_list()['glm3']

    @lazy_inline
    def __init__(self, config: ChatGLM2Config, **kwargs):
        super(ChatGLM2ForConditionalGeneration, self).__init__(config, **kwargs)
        self.transformer = ChatGLM2Model(config=config)
        self.cast = P.Cast()
        self.gather = P.Gather()
        dp = config.parallel_config.data_parallel
        cp = config.parallel_config.context_parallel
        mp = config.parallel_config.model_parallel
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        if config.parallel_config.vocab_emb_dp or (config.vocab_size % (dp * mp * cp) != 0):
            if loss_parallel_config.use_seq_parallel:
                if config.seq_length >= SHARDING_SEQ_LEN_THRESHOLD:
                    loss_parallel_config.data_parallel = dp * cp * mp
                    loss_parallel_config.model_parallel = 1
                    loss_parallel_config.context_parallel = 1
            else:
                loss_parallel_config.data_parallel = dp * cp
                loss_parallel_config.model_parallel = 1
                loss_parallel_config.context_parallel = 1
        else:
            loss_parallel_config.data_parallel = 1
            loss_parallel_config.model_parallel = dp * cp * mp
            loss_parallel_config.context_parallel = 1
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
                                     calculate_per_token_loss=calculate_per_token_loss,
                                     seq_split_num=config.parallel_config.seq_split_num)
        self.gmask = config.gmask_token_id
        self.bos_token_id = config.bos_token_id
        self.use_past = config.use_past
        self.rl_config = config.rl_config
        self.is_first_iteration = True
        self.not_equal = P.NotEqual()
        self.add = P.Add()
        self.reshape = P.Reshape()
        self.load_checkpoint(config)
        self.vocab_size = config.padded_vocab_size
        self.predict_run_mode = get_predict_run_mode()
        self.sub_batch_valid_len = P.Sub()
        enable_zero3 = config.parallel_config.recompute.parallel_optimizer_comm_recompute
        if enable_zero3:
            self._parallel_optimizer_comm_recompute(enable_zero3)

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get ChatGLM2 model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs, seq = input_ids.shape[0], input_ids.shape[1]
        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        batch_valid_length = Tensor(np.array([seq] * bs), mstype.int32)
        return input_ids, labels, None, None, None, None, None, batch_valid_length, None, None, slot_mapping, None, \
            None, None

    def set_dynamic_inputs(self, **kwargs):
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        if self.use_past:
            dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
            dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
            dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None, dynamic_batch_valid_length,
                            None, dynamic_block_tables, dynamic_slot_mapping, None, None, None)
        else:
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            None, None, None, None, None, None, None)
        logger.info("Set dynamic input for glm.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.transformer.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.transformer.encoder.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attention.infer_attention.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attention.infer_attention.rotary_embedding.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids=None, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, prefix_key_values=None,
                  block_tables=None, slot_mapping=None, batch_index=None, zactivate_len=None, input_mask=None):
        """ChatGLM2 for conditional generation model."""
        # input_ids: (bs, seq_len)
        # position_ids: (bs, seq_len)
        # attention_mask: (bs, seq_len)
        bs, seq_len = input_ids.shape
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
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
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            batch_valid_length = mint.cumsum(batch_valid_length, 0)
            hidden_states = self.gather(hidden_states, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        lm_logits = self.transformer.output_layer(hidden_states)
        outputs = (lm_logits,)
        if self.rl_config is not None:
            return lm_logits

        # train
        if labels is not None:
            logits = lm_logits.to(mstype.float32)
            labels = labels.reshape((-1,))
            logits = logits.reshape((-1, logits.shape[-1]))
            if input_mask is None:
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
        if not self.training and self.predict_run_mode:
            lm_logits = self.cast(lm_logits, mstype.float32)
            lm_logits = self.reshape(lm_logits, (-1, lm_logits.shape[-1]))
            outputs = (lm_logits,)

        return outputs

    @classmethod
    def convert_name(cls, weight_name):
        """convert HuggingFace weight name to MindFormers weight name"""
        origin_name = weight_name
        weight_name = weight_name.replace('.word_embeddings.weight', '.embedding_weight')
        weight_name = weight_name.replace('model.', 'transformer.encoder.')
        weight_name = weight_name.replace('.encoder.embed_tokens.weight', '.embedding.embedding_weight')
        weight_name = weight_name.replace('.down_proj.', '.dense_4h_to_h.')
        weight_name = weight_name.replace('.gate_up_proj.', '.dense_h_to_4h.')
        weight_name = weight_name.replace('.self_attn.q_proj.', '.self_attention.wq.')
        weight_name = weight_name.replace('.self_attn.k_proj.', '.self_attention.wk.')
        weight_name = weight_name.replace('.self_attn.v_proj.', '.self_attention.wv.')
        weight_name = weight_name.replace('.self_attn.o_proj.', '.self_attention.dense.')
        weight_name = weight_name.replace('.norm.', '.final_layernorm.')
        weight_name = weight_name.replace('lm_head.', 'transformer.output_layer.')
        if weight_name == origin_name:
            logger.warning(f"weight name '{weight_name}' does not change after conversion. "
                           f"Please check if it is as expected.")
        return weight_name

    @classmethod
    def convert_weight_dict(cls, source_dict, **kwargs):
        """convert HuggingFace weight dict to MindFormers weight dict"""
        model_config = kwargs.get("model_config")
        qkv_concat = model_config.qkv_concat if 'qkv_concat' in dir(model_config) else False
        target_dict = {}
        wq_keys = []
        wk_keys = []
        wv_keys = []
        w_qkv_keys = []
        w13_keys = []

        has_qkv_weights = _check_hf_qkv_weight(source_dict)
        for k, v in source_dict.items():
            k = cls.convert_name(k)
            target_dict.update({k: v})
            part = k.split('.')
            if part[-2] == 'wq':
                wq_keys.append(k)
            if part[-2] == 'wk':
                wk_keys.append(k)
            if part[-2] == 'wv':
                wv_keys.append(k)
            if part[-2] == "query_key_value":
                w_qkv_keys.append(k)
            if part[-2] == "dense_h_to_4h":
                w13_keys.append(k)

        if has_qkv_weights:
            if not qkv_concat:
                _split_qkv_weight(w_qkv_keys, model_config, target_dict)
                _split_ffn_weight(w13_keys, model_config, target_dict)
        else:
            if qkv_concat:
                qkv_dict = kwargs.get('qkv_dict', None)
                if not isinstance(qkv_dict, DictProxy):
                    raise ValueError(f'qkv_queue must be a queue, but got {qkv_dict}.')
                condition = kwargs.get('condition', None)
                if not isinstance(condition, Condition):
                    raise ValueError(f'condition must be a Condition, but got {condition}.')
                _concat_qkv_weight(wq_keys, wk_keys, wv_keys, qkv_dict, condition, target_dict)
            if not qkv_concat:
                _split_ffn_weight(w13_keys, model_config, target_dict)

        return target_dict

    @classmethod
    def convert_map_dict(cls, source_dict, **kwargs):
        """convert HuggingFace map dict to MindFormers map dict"""
        qkv_concat = kwargs.pop("qkv_concat", False)
        if not qkv_concat and check_safetensors_addition_param_support():
            logger.warning(f'When MS version is >= 2.6.0, setting `qkv_concat` is False may cause precision issue '
                           f'in current model {cls.__name__} after online hf weight conversion.')
        target_dict = {}
        wq_keys = []
        w_qkv_keys = []
        w13_keys = []
        has_qkv_weights = _check_hf_qkv_weight(source_dict)

        for k, v in source_dict.items():
            k = cls.convert_name(k)
            target_dict.update({k: v})
            part = k.split('.')
            if part[-2] == 'wq':
                wq_keys.append(k)
            if part[-2] == "query_key_value":
                w_qkv_keys.append(k)
            if part[-2] == "dense_h_to_4h":
                w13_keys.append(k)

        if has_qkv_weights:
            if not qkv_concat:
                for w_qkv_key in w_qkv_keys:
                    wq_key = w_qkv_key.replace('query_key_value', 'wq')
                    wk_key = w_qkv_key.replace('query_key_value', 'wk')
                    wv_key = w_qkv_key.replace('query_key_value', 'wv')
                    w_qkv_value = target_dict.pop(w_qkv_key)
                    target_dict.update({wq_key: w_qkv_value, wk_key: w_qkv_value, wv_key: w_qkv_value})
                for w13_key in w13_keys:
                    w1_key = w13_key.replace('dense_h_to_4h', 'dense_left')
                    w3_key = w13_key.replace('dense_h_to_4h', 'dense_right')
                    w13_value = target_dict.pop(w13_key)
                    target_dict.update({w1_key: w13_value, w3_key: w13_value})
        else:
            if qkv_concat:
                for wq_key in wq_keys:
                    wk_key = wq_key.replace('wq', 'wk')
                    wv_key = wq_key.replace('wq', 'wv')
                    wq_value = target_dict.pop(wq_key)
                    target_dict.pop(wk_key)
                    target_dict.pop(wv_key)

                    w_qkv_key = wq_key.replace('wq', 'query_key_value')
                    w_qkv_value = wq_value
                    target_dict.update({w_qkv_key: w_qkv_value})
            if not qkv_concat:
                for w13_key in w13_keys:
                    w13_value = target_dict.pop(w13_key)

                    w1_key = w13_key.replace('dense_h_to_4h', 'dense_left')
                    w3_key = w13_key.replace('dense_h_to_4h', 'dense_right')
                    target_dict.update({w1_key: w13_value, w3_key: w13_value})

        return target_dict

    @classmethod
    def obtain_qkv_ffn_concat_keys(cls):
        qkv_key = "query_key_value"
        ffn_key = "dense_h_to_4h"
        concat_keys = [qkv_key, ffn_key]
        logger.info(f"{cls.__name__} qkv/ffn concat keys are {concat_keys}")
        return concat_keys

    @classmethod
    def obtain_name_map(cls, load_checkpoint_files):
        name_map = dict()
        for checkpoint_file in load_checkpoint_files:
            with safe_open(checkpoint_file, framework="np") as f:
                for k in f.keys():
                    name_map.update({cls.convert_name(k): k})
        return name_map

    def clear_kv_cache(self):
        return self.transformer.clear_kv_cache()


def _check_hf_qkv_weight(source_dict):
    """check whether qkv weight exists in hf original weights"""
    for k, _ in source_dict.items():
        if "query_key_value" in k:
            return True
    return False


def _concat_qkv_weight(wq_keys, wk_keys, wv_keys, qkv_dict, condition, target_dict):
    """concat qkv weight from dicts"""
    # pop extra weight to shared dict if there is no corresponding weight for concat in the target dict
    for wk_key in wk_keys:
        wq_key = wk_key.replace('wk', 'wq')
        if wq_key not in wq_keys:
            with condition:
                qkv_dict[wk_key] = target_dict.pop(wk_key)
                condition.notify_all()
    for wv_key in wv_keys:
        wq_key = wv_key.replace('wv', 'wq')
        if wq_key not in wq_keys:
            with condition:
                qkv_dict[wv_key] = target_dict.pop(wv_key)
                condition.notify_all()

    # concat qkv
    for wq_key in wq_keys:
        wk_key = wq_key.replace('wq', 'wk')
        wv_key = wq_key.replace('wq', 'wv')
        wq_value = target_dict.pop(wq_key)
        wk_value = target_dict.pop(wk_key, None)
        wv_value = target_dict.pop(wv_key, None)

        # get missing weight from shared dict
        if wk_value is None:
            with condition:
                condition.wait_for(lambda: wk_key in qkv_dict.keys())
                wk_value = qkv_dict.pop(wk_key)
        if wv_value is None:
            with condition:
                condition.wait_for(lambda: wv_key in qkv_dict.keys())
                wv_value = qkv_dict.pop(wv_key)

        w_qkv_key = wq_key.replace('wq', 'query_key_value')
        w_qkv_value = np.concatenate((wq_value, wk_value, wv_value), 0)
        target_dict.update({w_qkv_key: w_qkv_value})


def _split_qkv_weight(w_qkv_keys, model_config, target_dict):
    """split qkv weight from dicts"""
    head_dim = model_config.kv_channels
    num_attention_heads = model_config.num_attention_heads
    n_kv_heads = model_config.multi_query_group_num or num_attention_heads
    hidden_size = head_dim * num_attention_heads
    kv_hidden_size = head_dim * n_kv_heads

    # split qkv
    for w_qkv_key in w_qkv_keys:
        w_qkv_value = target_dict.pop(w_qkv_key)
        qkv_dim = len(w_qkv_value.shape)
        if qkv_dim == 1:
            w_qkv_value = w_qkv_value.reshape(w_qkv_value.shape[0], -1)
        wq_value = w_qkv_value[:hidden_size, :]
        wk_value = w_qkv_value[hidden_size:hidden_size + kv_hidden_size, :]
        wv_value = w_qkv_value[hidden_size + kv_hidden_size:hidden_size + 2 * kv_hidden_size, :]
        wq_key = w_qkv_key.replace('query_key_value', 'wq')
        wk_key = w_qkv_key.replace('query_key_value', 'wk')
        wv_key = w_qkv_key.replace('query_key_value', 'wv')

        if qkv_dim == 1:
            wq_value = wq_value.reshape(wq_value.shape[0],)
            wk_value = wk_value.reshape(wk_value.shape[0],)
            wv_value = wv_value.reshape(wv_value.shape[0],)

        if check_safetensors_addition_param_support():
            wq_value = Parameter(Tensor(wq_value), requires_grad=True)
            wk_value = Parameter(Tensor(wk_value), requires_grad=True)
            wv_value = Parameter(Tensor(wv_value), requires_grad=True)

        target_dict.update({wq_key: wq_value, wk_key: wk_value, wv_key: wv_value})


def _split_ffn_weight(w13_keys, model_config, target_dict):
    """split ffn weight from dicts"""
    ffn_hidden_size = model_config.ffn_hidden_size

    # split ffn
    for w13_key in w13_keys:
        w13_value = target_dict.pop(w13_key)
        w1_value = w13_value[:ffn_hidden_size, :]
        w3_value = w13_value[ffn_hidden_size:, :]
        w1_key = w13_key.replace('dense_h_to_4h', 'dense_left')
        w3_key = w13_key.replace('dense_h_to_4h', 'dense_right')

        if check_safetensors_addition_param_support():
            w1_value = Parameter(Tensor(w1_value), requires_grad=True)
            w3_value = Parameter(Tensor(w3_value), requires_grad=True)

        target_dict.update({w1_key: w1_value, w3_key: w3_value})


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

    # pylint: disable=W0613
    # pylint: disable=W0221
    def construct(self, input_ids=None, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, prefix_key_values=None,
                  block_tables=None, slot_mapping=None, batch_index=None, zactivate_len=None):
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
            prefix_key_values=prefix_key_values,
            block_tables=block_tables,
            slot_mapping=slot_mapping
        )

    def kvcache(self, layer_idx):
        key_cache = \
            self.transformer.encoder.layers[layer_idx].self_attention.infer_attention.paged_attention_mgr.key_cache
        value_cache = \
            self.transformer.encoder.layers[layer_idx].self_attention.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache
