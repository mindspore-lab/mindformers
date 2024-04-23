# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Telechat models' APIs."""
import copy
import mindspore.common.dtype as mstype

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import Tensor, nn
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.models.utils import cell_reuse
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.modules import KVCachePreprocess
from mindformers.modules.layers import Linear
from mindformers.modules.transformer.transformer import LowerTriangularMaskWithDynamic
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.tools.logger import logger
from mindformers.models.llama.llama_layer import LlamaRMSNorm, FreqsMgr
from research.telechat.telechat_config import TelechatConfig
from research.telechat.telechat_layer import TelechatEmbedding
from research.telechat.telechat_transformer import TelechatDecodeLayer

__all__ = ['TelechatModel', 'TelechatForCausalLM']


class TelechatPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TelechatConfig
    base_model_prefix = "telechat"


def layer_compute_dtype(layer, layer_id, offset, parallel_config, n_layers, select_recompute=False):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            layer(Cell) - Represents the transformer block
            parallel_config(dict) - Parallel Config
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(Union[int, List[int]]) - Means the layer_index needs a offset, if there are other modules in the net.
            n_layers(int) - The total layers used for the model.
    """
    pp_dis = max(int((n_layers + 1) / parallel_config.pipeline_stage), 1)
    if isinstance(offset, list):
        if len(offset) != parallel_config.pipeline_stage:
            raise ValueError(f"The length of `offset` {len(offset)} do not match "
                             "`pipeline stage` {parallel_config.pipeline_stage}.")
        i = min(layer_id // pp_dis, parallel_config.pipeline_stage - 1)
        offset_layer = offset[i]
    elif isinstance(offset, int):
        offset_layer = offset
    else:
        raise TypeError(f"`offset` must be `int` of list of `int`, but got {type(offset)}.")

    pp_id = min((layer_id + offset_layer) // pp_dis, parallel_config.pipeline_stage - 1)
    layer.pipeline_stage = pp_id

    # Used for optimizer's fusion tag
    dis = max(int((n_layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        layer.set_comm_fusion(2)
    else:
        layer.set_comm_fusion(int((layer_id + offset_layer) / dis) + 1)
    if isinstance(parallel_config.recompute, bool):
        if parallel_config.recompute and not select_recompute:
            layer.recompute()
    else:
        if parallel_config.recompute.recompute and not select_recompute:
            layer.recompute(
                recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)


class TelechatModel(TelechatPreTrainedModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TelechatDecoderLayer`]
    Args:
        config(TelechatConfig): the config of network

    Returns:
            output: Tensor, the output of Telechat decoderlayer
    """

    def __init__(self,
                 config: TelechatConfig = None):
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
        self.use_flash_attention = config.use_flash_attention
        if self.use_flash_attention:
            logger.info("Enable flash attention.")
        elif config.use_flash_attention:
            logger.info("Current MindSpore do not support flash attention.")

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
                                  extend_method=config.extend_method)
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention)
        self.tok_embeddings = TelechatEmbedding(vocab_table_size=config.vocab_size,
                                                embedding_size=config.hidden_size,
                                                param_init_type=config.param_init_type)
        self.layers = nn.CellList()
        for layer_id in range(config.num_layers):
            layer = TelechatDecodeLayer(config.batch_size,
                                        config.seq_length,
                                        layer_id,
                                        dim=config.hidden_size,
                                        n_heads=config.num_heads,
                                        n_kv_heads=config.n_kv_heads,
                                        hidden_dropout_prob=config.hidden_dropout_prob,
                                        attention_dropout_prob=config.attention_dropout_prob,
                                        intermediate_size=config.intermediate_size,
                                        ffn_dim_multiplier=config.ffn_dim_multiplier,
                                        norm_eps=config.rms_norm_eps,
                                        qkv_has_bias=config.qkv_has_bias,
                                        compute_dtype=config.compute_dtype,
                                        layernorm_compute_dtype=config.layernorm_compute_type,
                                        softmax_compute_dtype=config.softmax_compute_type,
                                        rotary_dtype=config.rotary_dtype,
                                        param_init_type=config.param_init_type,
                                        use_past=config.use_past,
                                        use_flash_attention=config.use_flash_attention,
                                        is_dynamic=config.is_dynamic,
                                        use_kvcache_op=config.use_kvcache_op,
                                        is_flexible_shape=config.is_flexible_shape,
                                        use_rope_slice=config.use_rope_slice,
                                        parallel_config=config.parallel_config)
            layer_compute_dtype(layer, layer_id, config.offset, config.parallel_config,
                                config.num_layers, select_recompute=config.parallel_config.recompute.select_recompute)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type)
        self.kvcache_preprocess = KVCachePreprocess(max_batch_size=config.batch_size,
                                                    max_seq_length=config.seq_length,
                                                    is_dynamic=config.is_dynamic,
                                                    use_kvcache_op=config.use_kvcache_op,
                                                    is_flexible_shape=config.is_flexible_shape)

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
    def construct(self, tokens: Tensor, batch_valid_length=None, batch_index=None, zactivate_len=None):
        """
        Forward of telechat model.

        Args:
            tokens: the tokenized inputs with datatype int32
            input_position(Tensor): current position, used by model.predict.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.

        Returns:
            output: Tensor, the output of telechat decoderlayer
        """
        # preprocess
        bs, seq_len = self.shape(tokens)
        if not self.use_past:
            freqs_cis = self.freqs_mgr(seq_len)
            mask = self.casual_mask(tokens) # mask: [bs, seq, seq]
            kvcache_inputs = None
        else:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr.prefill(bs, seq_len)
                mask = self.casual_mask(tokens) # mask: [bs, seq, seq]
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)
                if self.is_dynamic and self.is_flexible_shape and not self.use_kvcache_op:
                    mask = self.casual_mask.increment_slice(
                        self.kvcache_preprocess.range,
                        self.kvcache_preprocess.max_cache_length // bs, batch_valid_length,
                        zactivate_len)
                else:
                    mask = self.casual_mask.increment(self.kvcache_preprocess.range, batch_valid_length, zactivate_len)
            kvcache_inputs = self.kvcache_preprocess(bs, batch_valid_length, batch_index, zactivate_len)

        # tokens: [bs, seq/1]
        h, embedding_weight = self.tok_embeddings(tokens)
        h = self.reshape(h, (bs, seq_len, self.hidden_size))
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            h = self.layers[i](h, freqs_cis, mask, kvcache_inputs=kvcache_inputs)
        output = self.norm_out(h)
        return output, embedding_weight

class TelechatHead(nn.Cell):
    """Head for Telechat to get the logits of each token in the vocab."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 compute_dtype="float16",
                 parallel_config=None):
        super(TelechatHead, self).__init__()
        copied_parallel_config = copy.deepcopy(parallel_config)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = compute_dtype
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        dp = copied_parallel_config.data_parallel
        mp = copied_parallel_config.model_parallel
        if parallel_config.vocab_emb_dp or (out_channels % mp != 0):
            self.matmul = P.MatMul(transpose_b=True).shard(((dp, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((dp, 1), (mp, 1)))

    def construct(self, x, embedding_weight=None):
        out_shape = P.Shape()(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        ori_dtype = F.dtype(x)
        weight = self.cast(embedding_weight, self.dtype)
        x = self.cast(x, self.dtype)
        x = self.matmul(x, weight)
        x = self.cast(x, ori_dtype)
        output = self.reshape(x, out_shape)
        return output

class TelechatForCausalLM(TelechatPreTrainedModel):
    r"""
        Provide telechat training loss or logits through network.

        Args:
            config (TelechatConfig): The config of telechat model.

        Returns:
            output: Tensor, the output of telechat decoderlayer
        """

    @cell_reuse
    def __init__(self, config: TelechatConfig = None):
        super(TelechatForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.config = config
        self.model_name = config.model_name
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
        self.logits_slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather(1)
        self.sub_batch_valid_len = P.Sub()
        self.model = TelechatModel(config=config)
        if self.model_name == 'telechat_12b':
            self.lm_head = Linear(in_channels=config.hidden_size,
                                  out_channels=config.vocab_size,
                                  has_bias=False,
                                  compute_dtype=config.compute_dtype,
                                  param_init_type=config.param_init_type,
                                  skip_redistribution=config.is_dynamic,
                                  weight_init="normal") # meta default: xavier_normal
        else:
            self.lm_head = TelechatHead(in_channels=config.hidden_size,
                                        out_channels=config.vocab_size,
                                        compute_dtype=config.compute_dtype,
                                        parallel_config=config.parallel_config)

        mp = config.parallel_config.model_parallel
        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        if vocab_size % mp != 0:
            logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config)
        self.seq_length = config.seq_length

        dp = config.parallel_config.data_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.slice.shard(((dp, 1),))
            self.logits_slice.shard(((dp, 1, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.mul.shard(((dp, 1), (dp, 1)))
            self.add.shard(((dp, 1), ()))
            self.gather.shard(((dp, 1, 1), (dp,)))
            self.sub_batch_valid_len.shard(((1,), ()))
            if self.model_name == 'telechat_12b':
                if config.parallel_config.vocab_emb_dp or (vocab_size % mp != 0):
                    self.lm_head.shard(strategy_matmul=((dp, 1), (1, 1)))
                else:
                    self.lm_head.shard(strategy_matmul=((dp, 1), (mp, 1)))
            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.load_checkpoint(config)

    # pylint: disable=W0613
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": Tensor(input_ids, mstype.int32)
        }

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None):
        r"""
        TelechatForCausalLM forward.

        Args:
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
            Tensor: The loss or (logits, tokens, input_mask) of the network.
        """
        bsz, seqlen = self.shape(input_ids)
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)

        tokens = input_ids
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        if not self.is_first_iteration:
            batch_valid_length = self.sub_batch_valid_len(batch_valid_length, 1)
        output, embedding_weight = self.model(tokens, batch_valid_length, batch_index, zactivate_len)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        if self.model_name == 'telechat_12b':
            logits = self.lm_head(output)
        else:
            logits = self.lm_head(output, embedding_weight)
        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is not None:
            input_mask = labels
        labels = input_ids
        if not self.training:
            if not pre_gather:
                logits = self.reshape(logits, (bsz, seqlen, -1))
            logits = self.cast(logits, mstype.float32)
            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)
            return logits, tokens, input_mask
        logits = self.logits_slice(logits, (0, 0, 0), (bsz, seqlen - 1, self.vocab_size), (1, 1, 1))
        labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
        input_mask = self.slice(input_mask, (0, 1), (bsz, seqlen), (1, 1))
        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss
