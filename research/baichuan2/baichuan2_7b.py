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
from mindformers.modules.transformer import AttentionMask, TransformerOpParallelConfig
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
        self.parallel_config = config.parallel_config
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.pad_token_id = config.pad_token_id
        self.slice = P.StridedSlice().shard(((1, 1),))
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.tok_embeddings = LlamaEmbedding(config.vocab_size, config.hidden_size,
                                                 param_init_type=config.param_init_type,
                                                 parallel_config=config.parallel_config)
            self.tok_embeddings.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.tok_embeddings.set_comm_fusion(2)
            else:
                self.tok_embeddings.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

            self.layers = nn.CellList()
            for layer_id in range(config.num_layers):
                layer = LLamaDecodeLayer(config.batch_size,
                                         config.seq_length,
                                         layer_id,
                                         dim=config.hidden_size,
                                         n_heads=config.num_heads,
                                         multiple_of=config.multiple_of,
                                         norm_eps=config.rms_norm_eps,
                                         compute_dtype=config.compute_dtype,
                                         layernorm_compute_dtype=config.layernorm_compute_type,
                                         softmax_compute_dtype=config.softmax_compute_type,
                                         rotary_dtype=config.rotary_dtype,
                                         param_init_type=config.param_init_type,
                                         use_past=config.use_past,
                                         compute_in_2d=config.compute_in_2d,
                                         parallel_config=config.parallel_config)
                layer_compute_dtype(layer, layer_id, config.offset,
                                    config.parallel_config, self.num_layers)
                self.layers.append(layer)

            self.norm_out = LlamaRMSNorm(
                config.hidden_size, config.rms_norm_eps,
                compute_type=config.layernorm_compute_type)

            self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
            if config.parallel_config.pipeline_stage > 1:
                self.norm_out.set_comm_fusion(2)
            else:
                self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            self.tok_embeddings = LlamaEmbedding(config.vocab_size, config.hidden_size,
                                                 param_init_type=config.param_init_type,
                                                 parallel_config=config.parallel_config)
            self.tok_embeddings.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.tok_embeddings.set_comm_fusion(2)
            else:
                self.tok_embeddings.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

            self.layers = nn.CellList()
            for layer_id in range(config.num_layers):
                layer = LLamaDecodeLayer(config.batch_size,
                                         config.seq_length,
                                         layer_id,
                                         dim=config.hidden_size,
                                         n_heads=config.num_heads,
                                         multiple_of=config.multiple_of,
                                         norm_eps=config.rms_norm_eps,
                                         compute_dtype=config.compute_dtype,
                                         layernorm_compute_dtype=config.layernorm_compute_type,
                                         softmax_compute_dtype=config.softmax_compute_type,
                                         rotary_dtype=config.rotary_dtype,
                                         param_init_type=config.param_init_type,
                                         use_past=config.use_past,
                                         compute_in_2d=config.compute_in_2d,
                                         parallel_config=config.parallel_config)
                layer_compute_dtype(layer, layer_id, config.offset,
                                    config.parallel_config, self.num_layers)
                self.layers.append(layer)

            self.norm_out = LlamaRMSNorm(
                config.hidden_size, config.rms_norm_eps,
                compute_type=config.layernorm_compute_type)
            if config.parallel_config.pipeline_stage > 1:
                self.norm_out.set_comm_fusion(2)
            else:
                self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
            if config.compute_in_2d:
                self.norm_out.shard(((config.parallel_config.data_parallel, 1),))
            else:
                self.norm_out.shard(((config.parallel_config.data_parallel, 1, 1),))
            self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
            if config.parallel_config.pipeline_stage > 1:
                self.norm_out.set_comm_fusion(2)
            else:
                self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.freqs_cos, self.freqs_sin, self.swap_mask = precompute_freqs_cis(
            config.hidden_size // config.num_heads, config.seq_length, dtype=config.rotary_dtype,
            pretrain_seqlen=config.pretrain_seqlen, extend_method=config.extend_method)
        self.get_attention_mask = AttentionMask(
            config.seq_length, parallel_config=config.parallel_config.dp_mp_config).to_float(config.compute_dtype)
        self.not_equal = P.NotEqual().shard(((config.parallel_config.data_parallel, 1), ()))
        self.freqs_size = config.hidden_size // config.num_heads

        # used for increased predict
        self.gather = P.Gather().shard(((1, 1), (1,)))
        # when in train process,it's always True;when in predict process,only first iteration is True.
        self.is_first_iteration = True
        self.all_ones_attention_mask = P.Ones()((1, 1, 1), mstype.float32)
        self.use_past = config.use_past
        self.input_position_delta = Tensor(np.arange(0, config.batch_size), mstype.int32) * config.seq_length
        self.sub = P.Sub().shard(((1,), (1,)))
        self.tile = P.Tile().shard(((1, 1, 1),))

    def construct(self, input_ids: Tensor, input_position=None, init_reset=True, batch_valid_length=None):
        """Forward of baichuan2_7b model."""
        bs, seq_len = input_ids.shape
        # (b, t, d) , dp, 1, 1
        h = self.tok_embeddings(input_ids)

        mask = None
        if self.is_first_iteration is False:
            # for increase predict
            input_position = self.sub(input_position, self.input_position_delta)
            freqs_cis = (self.reshape(self.gather(self.freqs_cos, input_position, 0), (bs, 1, seq_len, -1)),
                         self.reshape(self.gather(self.freqs_sin, input_position, 0), (bs, 1, seq_len, -1)),
                         self.swap_mask)
            mask = self.tile(self.all_ones_attention_mask, (bs, 1, 1))
        else:
            # first iteration of predict; all iterations of train
            freqs_cis = (self.tile(self.reshape(self.freqs_cos, (1, 1, seq_len, -1)), (bs, 1, 1, 1)),
                         self.tile(self.reshape(self.freqs_sin, (1, 1, seq_len, -1)), (bs, 1, 1, 1)),
                         self.swap_mask)
            input_mask = self.cast(self.not_equal(input_ids, self.pad_token_id), mstype.float32)
            mask = self.get_attention_mask(input_mask)

        # dp,1,1 -> dp,1,1
        for i in range(self.num_layers):
            h, _ = self.layers[i](h, freqs_cis, mask, init_reset=init_reset, batch_valid_length=batch_valid_length)
        # dp,1,1 -> dp,1,1
        output = self.norm_out(h)
        return output


class NormHead(nn.Cell):
    """
    NormHead Layer.

        Args:
            hidden_size (int): The hidden size of the input.
            vocab_size (int): Size of the dictionary of embeddings.
            compute_type (dtype.Number): The compute type.
            parallel_config (TransformerOpParallelConfig): The parallel config of network.

        Inputs:
            - hidden_states (Tensor) - Tensor of shape :math:`(batch, seq_length, hidden_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, vocab_size)`.
    """
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 compute_dtype=mstype.float32,
                 eps=1e-5,
                 parallel_config=TransformerOpParallelConfig()):
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
        self.eps = Tensor([eps], mstype.float32)

        self.matmul = P.MatMul(transpose_b=True)
        self.cast = P.Cast()
        self.compute_dtype = compute_dtype
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        if parallel_config.vocab_emb_dp:
            self.square.shard(((1, 1),))
            self.sqrt.shard(((1, 1),))
            self.add.shard(((1, 1), (1,)))
            self.real_div.shard(((1, 1), (1, 1)))
            self.matmul.shard(((parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.square.shard(((parallel_config.model_parallel, 1),))
            self.sqrt.shard(((parallel_config.model_parallel, 1),))
            self.add.shard(((parallel_config.model_parallel, 1), (1,)))
            self.real_div.shard(((parallel_config.model_parallel, 1),
                                 (parallel_config.model_parallel, 1)))
            self.matmul.shard(((parallel_config.data_parallel, 1),
                               (parallel_config.model_parallel, 1)))

        if parallel_config.pipeline_stage > 1:
            self.matmul.pipeline_stage = parallel_config.pipeline_stage - 1

    def construct(self, hidden_states):
        """Forward process of the NormHead"""
        out_shape = P.Shape()(hidden_states)[:-1] + (self.vocab_size,)
        hidden_states = P.Reshape()(hidden_states, (-1, self.hidden_size))

        variance = self.square(self.weight).sum(axis=1).reshape(-1, 1)
        variance_eps = self.sqrt(self.add(variance, self.eps))
        norm_weight = self.real_div(self.weight, variance_eps)

        ori_type = hidden_states.dtype
        out = self.matmul(hidden_states.astype(self.compute_dtype),
                          norm_weight.astype(self.compute_dtype))
        out = P.Reshape()(out, out_shape)
        return self.cast(out, ori_type)


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
        self.model = Baichuan7BV2Model(config=config)
        self.lm_head = NormHead(hidden_size=config.hidden_size,
                                vocab_size=config.vocab_size,
                                compute_dtype=config.compute_dtype,
                                parallel_config=config.parallel_config)

        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        parallel_config = config.parallel_config
        self.loss = CrossEntropyLoss(parallel_config=parallel_config)
        dp = parallel_config.data_parallel
        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.mul = P.Mul().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
        self.add = P.Add().shard(((parallel_config.data_parallel, 1), ()))

        # used for increased predict
        self.is_first_iteration = True

        self.load_checkpoint(config)

    # pylint: disable=W0613
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": Tensor(input_ids, mstype.int32)
        }

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None):
        """Baichuan7BV2ForCausalLM forward."""
        bsz, seqlen = input_ids.shape
        if self.phase == "train":
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids

        output = self.model(tokens, input_position, init_reset, batch_valid_length)
        logits = self.lm_head(output)

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
            label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
            input_mask = self.mul(input_mask, label_mask)

        logits = self.cast(logits, mstype.float32)
        if self.phase != "train":
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
