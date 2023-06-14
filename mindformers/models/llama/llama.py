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
"""LLaMA models' APIs."""

import mindspore.common.dtype as mstype
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import nn
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.base_model import BaseModel
from mindformers.modules.layers import Linear
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.modules.transformer.transformer import AttentionMask
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.pet.tuners.pet_adapter import PetAdapter
from mindformers.pet.tuners.lora_adapter import LoraAdapter

from .llama_config import LlamaConfig
from .llama_layer import LlamaEmbedding, LlamaRMSNorm, precompute_freqs_cis
from .llama_transformer import LLamaDecodeLayer

__all__ = ['LlamaModel', 'LlamaForCausalLM', 'LlamaForCausalLMWithLora']

def layer_compute_dtype(layer, layer_id, offset, parallel_config, n_layers):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            layer(Cell) - Represents the transformer block
            parallel_config(dict) - Parallel Config
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            n_layers(int) - The total layers used for the model.
    """
    pp_dis = max(int((n_layers + 1) / parallel_config.pipeline_stage), 1)
    pp_id = min((layer_id + offset) // pp_dis,
                parallel_config.pipeline_stage - 1)
    layer.pipeline_stage = pp_id

    # Used for optimizer's fusion tag
    dis = max(int((n_layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        layer.set_comm_fusion(2)
    else:
        layer.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    if isinstance(parallel_config.recompute, bool):
        if parallel_config.recompute:
            layer.recompute()
    else:
        if parallel_config.recompute.recompute:
            layer.recompute(
                recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)


class LlamaModel(BaseModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config(LlamaConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32

    Returns:
        output: Tensor, the output of llama decoderlayer
    """
    _support_list = MindFormerBook.get_model_support_list()['llama']

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
                                         n_heads=config.num_layers,
                                         multiple_of=config.multiple_of,
                                         norm_eps=config.rms_norm_eps,
                                         compute_dtype=config.compute_dtype,
                                         layernorm_compute_dtype=config.layernorm_compute_type,
                                         softmax_compute_dtype=config.softmax_compute_type,
                                         param_init_type=config.param_init_type,
                                         use_past=config.use_past,
                                         parallel_config=config.parallel_config)
                layer_compute_dtype(layer, layer_id, config.offset,
                                    config.parallel_config, self.num_layers)
                self.layers.append(layer)

            self.norm_out = LlamaRMSNorm(
                config.hidden_size, config.rms_norm_eps,
                param_init_type=config.param_init_type).to_float(config.layernorm_compute_type)

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
                                         param_init_type=config.param_init_type,
                                         use_past=config.use_past,
                                         parallel_config=config.parallel_config)
                layer_compute_dtype(layer, layer_id, config.offset,
                                    config.parallel_config, self.num_layers)
                self.layers.append(layer)

            self.norm_out = LlamaRMSNorm(
                config.hidden_size, config.rms_norm_eps,
                param_init_type=config.param_init_type).to_float(config.layernorm_compute_type)
            if config.parallel_config.pipeline_stage > 1:
                self.norm_out.set_comm_fusion(2)
            else:
                self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
            self.norm_out.shard(((config.parallel_config.data_parallel, 1, 1),))
            self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
            if config.parallel_config.pipeline_stage > 1:
                self.norm_out.set_comm_fusion(2)
            else:
                self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.freqs_cos, self.freqs_sin, self.mins_mask, self.rotary_mask = precompute_freqs_cis(
            config.hidden_size // config.num_heads, config.seq_length, dtype=config.compute_dtype
        )
        self.get_attention_mask = AttentionMask(
            config.seq_length, parallel_config=config.parallel_config.dp_mp_config).to_float(config.compute_dtype)
        self.not_equal = P.NotEqual().shard(((config.parallel_config.data_parallel, 1), ()))
        self.freqs_size = config.hidden_size // config.num_heads

    def construct(self, input_ids: Tensor):
        """Forward of llama model."""
        _, seqlen = input_ids.shape
        # (b, t, d) , dp, 1, 1
        h = self.tok_embeddings(input_ids)
        freqs_cis = (self.freqs_cos, self.freqs_sin, self.mins_mask, self.rotary_mask)

        mask = None
        if seqlen > 0:
            input_mask = self.cast(self.not_equal(input_ids, self.pad_token_id), mstype.float32)
            mask = self.get_attention_mask(input_mask)
        # dp,1,1 -> dp,1,1
        for i in range(self.num_layers):
            h, _ = self.layers[i](h, freqs_cis, mask)
        # dp,1,1 -> dp,1,1
        output = self.norm_out(h)
        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlamaForCausalLM(BaseModel):
    r"""
        Provide llama training loss or logits through network.
        Args:
            config (LlamaConfig): The config of llama model.

        Inputs:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            label_ids(Tensor): the tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`
            input_position(Tensor): current position, used by model.predict
            attention_mask(Tensor): Reserved param, not used.
            batch_valid_length(Tensor): Reserved param, not used.

        Returns:
            Tensor, the loss or logits of the network.

        Examples:
            >>> from mindformers.models.llama import LlamaConfig, LlamaForCausalLM
            >>> config = LlamaConfig(batch_size=2)
            >>> network = LlamaForCausalLM(config=config)
        """
    _support_list = MindFormerBook.get_model_support_list()['llama']

    def __init__(self, config: LlamaConfig = None):
        super(LlamaForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.model = LlamaModel(config=config)
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.lm_head = Linear(in_channels=config.hidden_size,
                                  out_channels=config.vocab_size,
                                  has_bias=False,
                                  compute_dtype=config.compute_dtype,
                                  param_init_type=config.param_init_type,
                                  weight_init="normal") # meta default: xavier_normal
            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1
        else:
            self.lm_head = Linear(in_channels=config.hidden_size,
                                  out_channels=config.vocab_size,
                                  has_bias=False,
                                  compute_dtype=config.compute_dtype,
                                  param_init_type=config.param_init_type,
                                  weight_init="normal") # meta default: xavier_normal
            if config.parallel_config.vocab_emb_dp:
                self.lm_head.shard(strategy_matmul=((config.parallel_config.data_parallel, 1), (1, 1)))
            else:
                self.lm_head.shard(strategy_matmul=((config.parallel_config.data_parallel, 1),
                                                    (config.parallel_config.model_parallel, 1)))
            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        parallel_config = config.parallel_config
        self.loss = CrossEntropyLoss(parallel_config=parallel_config)
        dp = parallel_config.data_parallel
        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.mul = P.Mul()
        self.gather = P.Gather().shard(((parallel_config.data_parallel, 1), (1,)))
        self.add = P.Add().shard(((parallel_config.data_parallel, 1), ()))
        self.load_checkpoint(config)

    # pylint: disable=W0613
    def construct(self,
                  input_ids,
                  label_ids=None,
                  input_position=None,
                  attention_mask=None,
                  batch_valid_length=None):
        """LlamaForCausalLM forward."""
        bsz, seqlen = input_ids.shape
        if self.phase == "train":
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids
        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if label_ids is None:
            label_ids = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            label_ids = self.slice(label_ids, (0, 1), (bsz, seqlen), (1, 1))
            label_mask = self.cast(self.not_equal(label_ids, self.ignore_token_id), mstype.float32)
            input_mask = self.mul(input_mask, label_mask)

        output = self.model(tokens)
        if input_position is not None:
            # predict
            if output.ndim > 2:
                output = self.reshape(output, (-1, output.shape[-1]))
            output = self.gather(output, input_position, 0)

            logits = self.lm_head(output)  # only compute last logits
        else:
            tokens = input_ids
            logits = self.lm_head(output)

        logits = self.cast(logits, mstype.float32)

        if self.phase != "train":
            logits = self.reshape(logits, (bsz, seqlen, -1))

            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        label_ids = self.reshape(label_ids, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, label_ids, input_mask)
        return loss

@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlamaForCausalLMWithLora(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig = None, pet=None):
        super().__init__(config)
        # get Pet tuning model.
        self.pet = pet
        self.pet.pet_config.reg_rules = r'.*wq|.*wv'
        self.model = LoraAdapter.get_pet_model(self.model, self.pet.pet_config)
        # freeze pretrained model
        PetAdapter.freeze_pretrained_model(self, self.pet.pet_type)
