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
"""Baichuan2_7b models' APIs."""
import math
import copy
import numpy as np
import mindspore.common.dtype as mstype

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import Tensor, nn, ops
from mindspore.context import ParallelMode
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.common.initializer import initializer, HeUniform
try:
    # pylint: disable=W0611
    from mindspore.nn.layer.flash_attention import FlashAttention
    FLASHATTENTION_VALID = True
except ImportError:
    FLASHATTENTION_VALID = False
try:
    # pylint: disable=W0611
    from mindspore.ops.operations.nn_ops import PromptFlashAttention
    PROMPTFLASHATTENTION_VALID = True
except ImportError:
    PROMPTFLASHATTENTION_VALID = False

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.models.base_model import BaseModel
from mindformers.models.utils import cell_reuse
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister

from mindformers.models.llama.llama import layer_compute_dtype
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.llama.llama_layer import LlamaEmbedding, LlamaRMSNorm, FreqsMgr, CausalMask
from mindformers.models.llama.llama_transformer import LLamaDecodeLayer
from mindformers.tools.logger import logger
from mindformers.modules import KVCachePreprocess
from mindformers.version_control import check_valid_flash_attention, check_valid_paged_attention

__all__ = ['Baichuan7BV2ForCausalLM', 'Baichuan7BV2Model']


class Baichuan7BV2Model(BaseModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config(LlamaConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32

    Returns:
        output: Tensor, the output of llama decoderlayer
    """

    def __init__(self,
                 config: LlamaConfig = None):
        super().__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        if config.batch_size or config.use_past:
            Validator.check_positive_int(config.batch_size)
        self.dtype = config.compute_dtype
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.n_head = config.num_heads
        self.head_dim = self.hidden_size // self.n_head
        self.pad_token_id = config.pad_token_id
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.is_dynamic = config.is_dynamic
        self.use_kvcache_op = config.use_kvcache_op
        self.is_flexible_shape = config.is_flexible_shape
        # only support flash attention in train and prefill predict process.
        self.use_flash_attention = config.use_flash_attention and (not self.use_past) and \
            check_valid_flash_attention(FLASHATTENTION_VALID, 'FlashAttention')
        # only support prompt flash attention in prefill phase of inference.
        self.use_prompt_flash_attention = config.use_prompt_flash_attention and self.use_past and \
            check_valid_flash_attention(PROMPTFLASHATTENTION_VALID, 'PromptFlashAttention')
        # only support paged attention in decoding phase of inference.
        self.use_paged_attention = config.use_paged_attention and self.use_past and check_valid_paged_attention()

        self.shape = P.Shape()
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.gather = P.Gather()
        self.slice = P.StridedSlice()

        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=config.seq_length,
                                  max_position_embedding=config.max_position_embedding,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method,
                                  is_dynamic=config.is_dynamic)
        self.casual_mask = CausalMask(seq_length=config.seq_length,
                                      compute_type=config.compute_dtype,
                                      is_dynamic=config.is_dynamic,
                                      pad_token_id=config.pad_token_id,
                                      use_flash_attention=self.use_flash_attention,
                                      use_prompt_flash_attention=self.use_prompt_flash_attention)
        self.tok_embeddings = LlamaEmbedding(vocab_table_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init_type=config.param_init_type,
                                             parallel_optimizer=True)
        self.layers = nn.CellList()
        for layer_id in range(config.num_layers):
            layer = LLamaDecodeLayer(config.batch_size,
                                     config.seq_length,
                                     layer_id,
                                     dim=config.hidden_size,
                                     n_heads=config.num_heads,
                                     n_kv_heads=config.n_kv_heads,
                                     intermediate_size=config.intermediate_size,
                                     multiple_of=config.multiple_of,
                                     ffn_dim_multiplier=config.ffn_dim_multiplier,
                                     norm_eps=config.rms_norm_eps,
                                     qkv_has_bias=config.qkv_has_bias,
                                     qkv_concat=config.qkv_concat,
                                     compute_dtype=config.compute_dtype,
                                     layernorm_compute_dtype=config.layernorm_compute_type,
                                     softmax_compute_dtype=config.softmax_compute_type,
                                     rotary_dtype=config.rotary_dtype,
                                     param_init_type=config.param_init_type,
                                     use_past=config.use_past,
                                     use_flash_attention=self.use_flash_attention,
                                     use_prompt_flash_attention=self.use_prompt_flash_attention,
                                     use_paged_attention=self.use_paged_attention,
                                     block_size=config.block_size,
                                     num_blocks=config.num_blocks,
                                     is_dynamic=config.is_dynamic,
                                     use_kvcache_op=config.use_kvcache_op,
                                     is_flexible_shape=config.is_flexible_shape,
                                     use_rope_slice=config.use_rope_slice,
                                     parallel_config=config.parallel_config)
            layer_compute_dtype(layer, layer_id, config.offset, config.parallel_config,
                                config.num_layers, select_recompute=config.parallel_config.recompute.select_recompute)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type, is_dynamic=config.is_dynamic)
        self.kvcache_preprocess = KVCachePreprocess(max_batch_size=config.batch_size,
                                                    max_seq_length=config.seq_length,
                                                    is_dynamic=config.is_dynamic,
                                                    use_kvcache_op=config.use_kvcache_op,
                                                    is_flexible_shape=config.is_flexible_shape,
                                                    use_paged_attention=config.use_paged_attention)

        dp = config.parallel_config.data_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.tok_embeddings.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.tok_embeddings.set_comm_fusion(2)
                self.norm_out.set_comm_fusion(2)
            else:
                self.tok_embeddings.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
                self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

            self.tok_embeddings.shard(config.parallel_config)
            self.casual_mask.shard(config.parallel_config)
            self.norm_out.shard((dp, 1, 1))

    # pylint: disable=W0613
    def construct(self, tokens: Tensor, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None):
        """
        Forward of llama model.

        Args:
            tokens: the tokenized inputs with datatype int32
            input_position(Tensor): current position, used by model.predict.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            batch_index(Tensor): The generated batch index when use continuous batching in LLM serving.
                Tensor of shape :math:`(batch_size,)`. Default None.
            zactivate_len(Tensor): The slice length of KVCache when use dynamic shape infer.
                Tensor of shape :math:`(seq_length,)`. Default None.

        Returns:
            output: Tensor, the output of llama decoderlayer
        """
        # preprocess
        bs, seq_len = self.shape(tokens)
        if not self.use_past:
            freqs_cis = self.freqs_mgr()
            mask = self.casual_mask(tokens) # mask: [bs, seq, seq]
            kvcache_inputs = None
        else:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr(seq_len)
                mask = self.casual_mask(tokens) # mask: [bs, seq, seq]
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length, bs)
                if self.is_dynamic and self.is_flexible_shape and not self.use_kvcache_op:
                    mask = self.casual_mask.increment_slice(self.kvcache_preprocess.range,
                                                            self.kvcache_preprocess.max_cache_length // bs,
                                                            batch_valid_length, zactivate_len)
                else:
                    mask = self.casual_mask.increment(self.kvcache_preprocess.range, batch_valid_length, zactivate_len)

            kvcache_inputs = self.kvcache_preprocess(bs, batch_valid_length, batch_index, zactivate_len,
                                                     block_tables, slot_mapping)

        # tokens: [bs, seq/1]
        h = self.tok_embeddings(tokens)
        h = self.reshape(h, (bs, seq_len, self.hidden_size))
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            h = self.layers[i](h, freqs_cis, mask, kvcache_inputs=kvcache_inputs)
        output = self.norm_out(h)
        return output


class NormHead(nn.Cell):
    """
    NormHead Layer.

        Args:
            hidden_size (int): The hidden size of the input.
            vocab_size (int): Size of the dictionary of embeddings.
            compute_type (dtype.Number): The compute type.
            eps (number): A small positive value prevents division by zero.

        Inputs:
            - hidden_states (Tensor) - Tensor of shape :math:`(batch, seq_length, hidden_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, vocab_size)`.
    """
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 use_past,
                 is_dynamic=False,
                 compute_dtype=mstype.float16,
                 eps=1e-5):
        super().__init__()
        self.weight = Parameter(
            initializer(HeUniform(negative_slope=math.sqrt(5)),
                        [vocab_size, hidden_size],
                        mstype.float16),
            name='weight',
            parallel_optimizer=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.add = P.Add()
        self.real_div = P.RealDiv()
        self.reshape = P.Reshape()
        if is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.sum = P.ReduceSum()
        self.eps = Tensor([eps], mstype.float16)
        self.is_first_iteration = True
        self.use_past = use_past

        self.matmul = P.MatMul(transpose_b=True)
        self.cast = P.Cast()
        self.compute_dtype = compute_dtype
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.assign = P.Assign()

    def construct(self, hidden_states):
        """Forward process of the NormHead"""
        out_shape = P.Shape()(hidden_states)[:-1] + (self.vocab_size,)
        hidden_states = self.reshape(hidden_states, (-1, self.hidden_size))

        if self.is_first_iteration:
            variance = self.square(self.weight)
            variance = self.sum(variance, 1)
            variance = self.reshape(variance, (-1, 1))
            variance_eps = self.sqrt(self.add(variance, self.eps))
            norm_weight = self.real_div(self.weight, variance_eps)
            if self.use_past:
                norm_weight = ops.depend(norm_weight, norm_weight)
                self.assign(self.weight, norm_weight)
        else:
            norm_weight = self.weight
            self.assign(self.weight, norm_weight)
            norm_weight = ops.depend(norm_weight, norm_weight)

        ori_type = hidden_states.dtype
        out = self.matmul(hidden_states.astype(self.compute_dtype),
                          norm_weight.astype(self.compute_dtype))
        out = self.reshape(out, out_shape)
        return self.cast(out, ori_type)

    def shard(self, parallel_config):
        """sharding for norm head"""
        self.square.shard(((parallel_config.model_parallel * parallel_config.data_parallel, 1),))
        self.sqrt.shard(((parallel_config.model_parallel * parallel_config.data_parallel, 1),))
        self.add.shard(((parallel_config.model_parallel * parallel_config.data_parallel, 1), (1,)))
        self.real_div.shard(((parallel_config.model_parallel * parallel_config.data_parallel, 1),
                             (parallel_config.model_parallel * parallel_config.data_parallel, 1)))
        self.sum.shard(((parallel_config.model_parallel * parallel_config.data_parallel, 1),))
        self.matmul.shard(((1, 1),
                           (parallel_config.model_parallel * parallel_config.data_parallel, 1)))


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class Baichuan7BV2ForCausalLM(BaseModel):
    r"""
        Provide baichuan2_7b training loss or logits through network.
        Args:
            config (LlamaConfig): The config of baichuan2_7b model.

        Inputs:
            input_ids(Tensor): The tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            labels(Tensor): The tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            input_position(Tensor): Current position, used by model.predict.
            position_ids(Tensor): Reserved param, not used.
            attention_mask(Tensor): Reserved param, not used.
            input_embeds(Tensor): Reserved param, not used.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): The past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            batch_index(Tensor): The generated batch index when use continuous batching in LLM serving.
                Tensor of shape :math:`(batch_size,)`. Default None.
            zactivate_len(Tensor): The slice length of KVCache when use dynamic shape infer.
                Tensor of shape :math:`(seq_length,)`. Default None.

        Returns:
            Tensor, the loss or logits of the network.
        """

    @cell_reuse
    def __init__(self, config: LlamaConfig = None):
        super(Baichuan7BV2ForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.config = config
        self.seq_length = config.seq_length
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather(1)
        self.sub_batch_valid_len = P.Sub()
        self.model = Baichuan7BV2Model(config=config)
        self.lm_head = NormHead(hidden_size=config.hidden_size,
                                vocab_size=config.vocab_size,
                                use_past=config.use_past,
                                is_dynamic=config.is_dynamic,
                                compute_dtype=config.compute_dtype)

        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        loss_parallel_config.model_parallel = loss_parallel_config.model_parallel * loss_parallel_config.data_parallel
        loss_parallel_config.data_parallel = 1
        if vocab_size % (loss_parallel_config.model_parallel) != 0:
            logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, loss_parallel_config.model_parallel)
            logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config)

        dp = config.parallel_config.data_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.slice.shard(((dp, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.mul.shard(((dp, 1), (dp, 1)))
            self.add.shard(((dp, 1), ()))
            self.gather.shard(((dp, 1, 1), (dp,)))
            self.sub_batch_valid_len.shard(((1,), ()))
            self.lm_head.shard(config.parallel_config)

            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.lm_head.set_comm_fusion(2)
            else:
                self.lm_head.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.load_checkpoint(config)

    # pylint: disable=W0613
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": Tensor(input_ids, mstype.int32)
        }

    def prepare_inputs_for_export(self, full_model=True):
        """prepare_inputs_for_export"""
        dyn = self.config.is_dynamic
        use_paged_attention = self.config.use_paged_attention
        if dyn:
            logger.info(f"Exporting dynamic MindIR...")
        if use_paged_attention:
            logger.info(f"Exporting model with paged attention...")
        bs = None if dyn else self.config.batch_size
        seq_len = None if dyn else self.seq_length
        prefill_mapping_len = None if dyn else bs * seq_len
        inc_mapping_len = None if dyn else bs * 1
        block_size = self.config.block_size
        max_num_blocks_per_batch = None if dyn else self.seq_length // block_size

        def dummy_tensor(shape, dtype):
            if None in shape:
                return Tensor(shape=shape, dtype=dtype)
            return Tensor(np.ones(shape=tuple(shape)), dtype=dtype)

        batch_valid_length = dummy_tensor(shape=[bs], dtype=mstype.int32)
        batch_index = None if use_paged_attention else dummy_tensor(shape=[bs], dtype=mstype.int64)
        zactivate_len = None if use_paged_attention else dummy_tensor(shape=[seq_len], dtype=mstype.int64)

        if full_model:
            logger.info('\nexporting with batch_size = %s, seq = %s ...', self.config.batch_size, self.seq_length)
            input_ids = dummy_tensor(shape=[bs, seq_len], dtype=mstype.int32)
            block_tables = None
            slot_mapping = dummy_tensor(shape=[prefill_mapping_len],
                                        dtype=mstype.int32) if use_paged_attention else None
        else:
            logger.info('\nexporting with batch_size = %s, seq = 1 ...', self.config.batch_size)
            input_ids = dummy_tensor(shape=[bs, 1], dtype=mstype.int32)
            block_tables = dummy_tensor(shape=[inc_mapping_len, max_num_blocks_per_batch],
                                        dtype=mstype.int32) if use_paged_attention else None
            slot_mapping = dummy_tensor(shape=[inc_mapping_len],
                                        dtype=mstype.int32) if use_paged_attention else None
        return input_ids, None, None, None, None, None, None, batch_valid_length, batch_index, zactivate_len, \
            block_tables, slot_mapping

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None):
        """Baichuan7BV2 ForCausalLM forward."""
        bsz, seqlen = self.shape(input_ids)
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        if not self.is_first_iteration:
            batch_valid_length = self.sub_batch_valid_len(batch_valid_length, 1)
        output = self.model(tokens, batch_valid_length, batch_index, zactivate_len, block_tables, slot_mapping)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.lm_head(output)

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_mask, label_mask)

        if not self.training:
            if not pre_gather:
                logits = self.reshape(logits, (bsz, seqlen, -1))
            logits = self.cast(logits, mstype.float32)
            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss
