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
import numpy as np
import mindspore.common.dtype as mstype

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import Tensor, nn
from mindspore.context import ParallelMode
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.common.initializer import initializer, HeUniform

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.models.base_model import BaseModel
from mindformers.models.utils import cell_reuse
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.modules.transformer import AttentionMask
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister

from mindformers.models.llama.llama import layer_compute_dtype
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.llama.llama_layer import LlamaEmbedding, LlamaRMSNorm, precompute_freqs_cis
from mindformers.models.llama.llama_transformer import LLamaDecodeLayer

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
        self.num_layers = config.num_layers
        self.pad_token_id = config.pad_token_id
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.use_flash_attention = config.use_flash_attention and FLASHATTENTION_VALID
        if self.use_flash_attention:
            logger.info("Enable flash attention.")
        elif config.use_flash_attention:
            logger.info("Current MindSpore do not support flash attention.")

        self.freqs_cos, self.freqs_sin, self.swap_mask = precompute_freqs_cis(
            config.hidden_size // config.num_heads, config.seq_length, dtype=config.rotary_dtype,
            pretrain_seqlen=config.pretrain_seqlen, extend_method=config.extend_method)
        self.get_attention_mask = AttentionMask(
            config.seq_length, parallel_config=config.parallel_config.dp_mp_config).to_float(config.compute_dtype)
        self.multiply_data = Tensor([-10000.0], dtype=config.compute_dtype)
        self.one = Tensor([1.0], dtype=config.compute_dtype)
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.mul_mask = P.Mul()
        self.sub = P.Sub()
        self.expand_dims = P.ExpandDims()
        self.not_equal = P.NotEqual()
        self.gather = P.Gather()

        self.tok_embeddings = LlamaEmbedding(
            config.vocab_size, config.hidden_size, param_init_type=config.param_init_type)
        self.layers = nn.CellList()
        for layer_id in range(config.num_layers):
            layer = LLamaDecodeLayer(config.batch_size,
                                     config.seq_length,
                                     layer_id,
                                     dim=config.hidden_size,
                                     n_heads=config.num_heads,
                                     multiple_of=config.multiple_of,
                                     n_kv_heads=config.n_kv_heads,
                                     ffn_dim_multiplier=config.ffn_dim_multiplier,
                                     norm_eps=config.rms_norm_eps,
                                     compute_dtype=config.compute_dtype,
                                     layernorm_compute_dtype=config.layernorm_compute_type,
                                     softmax_compute_dtype=config.softmax_compute_type,
                                     rotary_dtype=config.rotary_dtype,
                                     param_init_type=config.param_init_type,
                                     use_past=config.use_past,
                                     use_flash_attention=config.use_flash_attention,
                                     compute_in_2d=config.compute_in_2d,
                                     use_past_shard=config.use_past_shard,
                                     parallel_config=config.parallel_config)
            layer_compute_dtype(layer, layer_id, config.offset, config.parallel_config,
                                config.num_layers, select_recompute=config.parallel_config.recompute.select_recompute)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type)

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

            self.tile.shard(((1, 1, 1, 1), ()))
            self.sub.shard(((1,), (dp, 1, 1)))
            self.mul_mask.shard(((dp, 1, 1, 1), (1,)))
            self.expand_dims.shard(((dp, 1, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.gather.shard(((dp, 1), (1,)))
            if config.compute_in_2d:
                self.norm_out.shard((dp, 1))
            else:
                self.norm_out.shard((dp, 1, 1))

        if self.use_past:
            seq_range = np.arange(config.seq_length).reshape(1, 1, -1)
            self.ones = P.Ones()
            self.range = Tensor(np.tile(seq_range, (config.batch_size, 1, 1)), mstype.int32)
            self.gather_past = P.Gather()
            self.expand_dims = P.ExpandDims()
            self.le_past = P.LessEqual()

    # pylint: disable=W0613
    def construct(self, tokens: Tensor, input_position=None, init_reset=True, batch_valid_length=None):
        """Forward of llama model."""
        # preprocess
        bs, seq_len = tokens.shape
        if self.use_past:
            if not isinstance(init_reset, Tensor):
                init_reset = Tensor([init_reset], mstype.bool_)
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bs, 1), mstype.int32)
        if self.is_first_iteration:
            freqs_cis = (self.tile(self.reshape(self.freqs_cos, (1, 1, seq_len, -1)), (bs, 1, 1, 1)),
                         self.tile(self.reshape(self.freqs_sin, (1, 1, seq_len, -1)), (bs, 1, 1, 1)),
                         self.swap_mask)
            input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), self.dtype)
            mask = self.get_attention_mask(input_mask)
            # mask: [bs, seq, seq]
        else:
            cur_pos = batch_valid_length - 1
            valid_length = self.reshape(cur_pos, (-1, 1, 1))
            freqs_cis = (self.reshape(self.gather_past(self.freqs_cos, cur_pos, 0), (bs, 1, seq_len, -1)),
                         self.reshape(self.gather_past(self.freqs_sin, cur_pos, 0), (bs, 1, seq_len, -1)),
                         self.swap_mask)
            mask = self.cast(self.le_past(self.range, valid_length), self.dtype)
            # mask: [bs, 1, 1]
        mask = self.sub(self.one, self.cast(mask, self.dtype))
        if not self.use_flash_attention:
            mask = self.expand_dims(mask, 1)
            mask = self.mul_mask(mask, self.multiply_data)

        # tokens: [bs, seq/1]
        h = self.tok_embeddings(tokens)
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            h, _ = self.layers[i](h, freqs_cis, mask,
                                  init_reset=init_reset, batch_valid_length=batch_valid_length)
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
                 compute_dtype=mstype.float32,
                 eps=1e-5):
        super().__init__()
        self.weight = Parameter(
            initializer(HeUniform(negative_slope=math.sqrt(5)),
                        [vocab_size, hidden_size],
                        mstype.float32),
            name='weight',
            parallel_optimizer=False)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.add = P.Add()
        self.real_div = P.RealDiv()
        self.reshape = P.Reshape()
        self.sum = P.ReduceSum()
        self.eps = Tensor([eps], mstype.float32)

        self.matmul = P.MatMul(transpose_b=True)
        self.cast = P.Cast()
        self.compute_dtype = compute_dtype
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def construct(self, hidden_states):
        """Forward process of the NormHead"""
        out_shape = P.Shape()(hidden_states)[:-1] + (self.vocab_size,)
        hidden_states = self.reshape(hidden_states, (-1, self.hidden_size))

        variance = self.square(self.weight)
        variance = self.sum(variance, 1)
        variance = self.reshape(variance, (-1, 1))
        variance_eps = self.sqrt(self.add(variance, self.eps))
        norm_weight = self.real_div(self.weight, variance_eps)

        ori_type = hidden_states.dtype
        out = self.matmul(hidden_states.astype(self.compute_dtype),
                          norm_weight.astype(self.compute_dtype))
        out = self.reshape(out, out_shape)
        return self.cast(out, ori_type)

    def shard(self, parallel_config):
        """sharding for norm head"""
        if parallel_config.vocab_emb_dp:
            self.square.shard(((1, 1),))
            self.sqrt.shard(((1, 1),))
            self.add.shard(((1, 1), (1,)))
            self.real_div.shard(((1, 1), (1, 1)))
            self.sum.shard(((1, 1),))
            self.matmul.shard(((parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.square.shard(((parallel_config.model_parallel, 1),))
            self.sqrt.shard(((parallel_config.model_parallel, 1),))
            self.add.shard(((parallel_config.model_parallel, 1), (1,)))
            self.real_div.shard(((parallel_config.model_parallel, 1),
                                 (parallel_config.model_parallel, 1)))
            self.sum.shard(((parallel_config.model_parallel, 1),))
            self.matmul.shard(((parallel_config.data_parallel, 1),
                               (parallel_config.model_parallel, 1)))


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class Baichuan7BV2ForCausalLM(BaseModel):
    r"""
        Provide baichuan2_7b training loss or logits through network.
        Args:
            config (LlamaConfig): The config of baichuan2_7b model.

        Inputs:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            labels(Tensor): the tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            input_position(Tensor): current position, used by model.predict.
            position_ids(Tensor): Reserved param, not used.
            attention_mask(Tensor): Reserved param, not used.
            input_embeds(Tensor): Reserved param, not used.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
              prediction. Tensor of shape :math:`(batch_size,)`. Default None.

        Returns:
            Tensor, the loss or logits of the network.
        """

    @cell_reuse()
    def __init__(self, config: LlamaConfig = None):
        super(Baichuan7BV2ForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.dtype = config.compute_dtype
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.model = Baichuan7BV2Model(config=config)
        self.lm_head = NormHead(hidden_size=config.hidden_size,
                                vocab_size=config.vocab_size,
                                compute_dtype=config.compute_dtype)
        self.loss = CrossEntropyLoss(parallel_config=config.parallel_config)

        dp = config.parallel_config.data_parallel
        self.slice.shard(((dp, 1),))
        self.not_equal.shard(((dp, 1), ()))
        self.mul.shard(((dp, 1), (dp, 1)))
        self.add.shard(((dp, 1), ()))
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

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None):
        """LlamaForCausalLM forward."""
        bsz, seqlen = input_ids.shape
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids

        output = self.model(tokens, input_position, init_reset, batch_valid_length)
        logits = self.lm_head(output)

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), self.dtype)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), self.dtype)
                input_mask = self.mul(input_mask, label_mask)

        logits = self.cast(logits, mstype.float32)
        if not self.training:
            logits = self.reshape(logits, (bsz, seqlen, -1))
            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss
