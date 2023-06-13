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

"""GPT model"""
import copy
import numpy as np

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindformers.modules.transformer.moe import default_moe_config
from mindformers.modules.layers import LayerNorm, Dropout
from mindformers.core.loss import CrossEntropyLoss
from mindformers.modules.transformer import AttentionMask, VocabEmbedding
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.base_model import BaseModel
from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.logger import logger
from .gpt2_config import GPT2Config
from .gpt_modules import GPTTransformerDecoderLayer

__all__ = ['GPT2LMHeadModel']


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GPT2LMHeadModel(BaseModel):
    r"""
        Provide gpt training loss or logits through network.
        Args:
            config (GPT2Config): The config of Gpt2Model.

        Returns:
            Tensor, the loss or logits of the network.
        """
    _support_list = MindFormerBook.get_model_support_list()['gpt2']

    def __init__(self, config: GPT2Config = None):
        config = config if config is not None else GPT2Config()
        super(GPT2LMHeadModel, self).__init__(config, auto_prefix=False)

        self.eos_token_id = self.config.eos_token_id
        parallel_config = self.config.parallel_config
        self.stridedslice = P.StridedSlice().shard(((parallel_config.data_parallel, 1),))
        self.not_equal = P.NotEqual().shard(((parallel_config.data_parallel, 1), ()))

        self.get_attention_mask = AttentionMask(seq_length=config.seq_length,
                                                parallel_config=parallel_config.dp_mp_config)

        self.backbone = GPT2Model(config)
        self.head = GPTHead(hidden_size=config.hidden_size,
                            vocab_size=config.vocab_size,
                            parallel_config=self.config.parallel_config)
        if parallel_config.pipeline_stage > 1:
            self.head.pipeline_stage = parallel_config.pipeline_stage - 1
            self.backbone.embedding.word_embedding.embedding_table.add_pipeline_stage(self.head.pipeline_stage)

        mp = config.parallel_config.model_parallel
        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(parallel_config)
        if vocab_size % mp != 0:
            logger.warning("The vocab size of GPT Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of GPT Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1

        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config)
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.load_checkpoint(config)
        self.add = P.Add().shard(((parallel_config.data_parallel, 1), ()))

    def construct(self, input_ids, input_mask=None):
        r"""
            construct function for Language Modeling

            Args:
                input_ids (Tensor): the indices of input sequence tokens in the vocabulary.
                input_mask (Tensor): input sentences padding mask, where 0 indicates padding position.

            Returns:
                logits (Tensor) or loss (mstype.float32): if is_training is False, directly return the logits,
                                                         otherwise, return the computed loss.
        """

        batch_size, seq_length = input_ids.shape

        if self.phase == "train":
            tokens = self.stridedslice(input_ids, (0, 0), (batch_size, seq_length - 1), (1, 1))
            if input_mask is not None:
                input_mask = self.stridedslice(input_mask, (0, 0), (batch_size, seq_length - 1), (1, 1))
            else:
                input_mask = self.not_equal(tokens, self.eos_token_id)
        else:
            tokens = input_ids
            if input_mask is None:
                input_mask = self.not_equal(tokens, self.eos_token_id)

        input_mask = self.cast(input_mask, mstype.float32)
        attention_mask = self.get_attention_mask(input_mask)

        # [batch_size, seq_length, vocab_size]
        output_states, embedding_table = self.backbone(tokens, attention_mask)
        logits = self.head(output_states, embedding_table)

        if self.phase != 'train':
            logits = self.reshape(logits, (batch_size, seq_length, -1))

            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)

            return logits, tokens, input_mask

        labels = self.stridedslice(input_ids, (0, 1), (batch_size, seq_length), (1, 1))
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))

        loss = self.loss(logits, labels, input_mask)
        return loss


class GPTEmbeddingLayer(nn.Cell):
    r"""The Embedding Layer of GPT-2 network."""

    def __init__(self, config: GPT2Config = None):
        super(GPTEmbeddingLayer, self).__init__()
        parallel_config = copy.deepcopy(config.parallel_config)
        embedding_mp = config.parallel_config.embedding_dp_mp_config.model_parallel
        vocab_size = config.vocab_size
        if vocab_size % embedding_mp != 0:
            logger.warning("The vocab size of embedding layer is: %s, it is not divide by model_parallel: %s",
                           vocab_size, embedding_mp)
            logger.warning("Now, model_parallel will be changed: mp = 1")
            parallel_config.embedding_dp_mp_config.model_parallel = 1

        self.word_embedding = VocabEmbedding(vocab_size=vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init=initializer('normal',
                                                                    [vocab_size, config.hidden_size],
                                                                    dtype=mstype.float32),
                                             parallel_config=parallel_config.embedding_dp_mp_config)
        new_parallel_config = copy.deepcopy(parallel_config)
        new_parallel_config.vocab_emb_dp = True

        self.position_embedding = VocabEmbedding(vocab_size=config.seq_length,
                                                 embedding_size=config.hidden_size,
                                                 param_init=initializer('normal',
                                                                        [config.seq_length, config.hidden_size],
                                                                        dtype=mstype.float32),
                                                 parallel_config=new_parallel_config.embedding_dp_mp_config)
        self.add = P.Add().shard(
            ((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
        self.dropout = Dropout(1 - config.embedding_dropout_prob)
        self.dropout.shard(((parallel_config.data_parallel, 1, 1),))

    def construct(self, input_ids, input_position):
        """The forward compute of Embedding Layer."""
        word_embedding, word_table = self.word_embedding(input_ids)
        position_embedding, _ = self.position_embedding(input_position)
        embedding = self.add(word_embedding, position_embedding)
        embedding = self.dropout(embedding)
        return embedding, word_table


def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            network(Cell) - Represents the transformer block
            parallel_config(dict) - Parallel Config
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
    """
    pp_dis = max(int((layers + 1) / parallel_config.pipeline_stage), 1)
    pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)
    network.pipeline_stage = pp_id

    # Used for optimizer's fusion tag
    dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        network.set_comm_fusion(2)
    else:
        network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    if isinstance(parallel_config.recompute, bool):
        if parallel_config.recompute:
            network.recompute()
    else:
        if parallel_config.recompute.recompute:
            network.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)


class GPT2Model(nn.Cell):
    """
    The backbone of GPT network

    Args:
        config(GPT2Config): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32
        input_mask: the mask indicating whether each position is a valid input

    Returns:
        output_state: Tensor, the output logit of backbone
        present_layer: Tensor, the current feature map
        embedding_table: Tensor, the embedding table for the vocabulary
    """

    def __init__(self, config):
        super(GPT2Model, self).__init__()

        self.embedding = GPTEmbeddingLayer(config)
        self.embedding.pipeline_stage = 0

        self.layernorm = LayerNorm((config.hidden_size,)).to_float(config.layernorm_compute_type)
        if config.parallel_config.pipeline_stage > 1:
            self.layernorm.set_comm_fusion(2)
        else:
            self.layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.layernorm.shard(((config.parallel_config.data_parallel, 1),))
        self.layernorm.pipeline_stage = config.parallel_config.pipeline_stage - 1

        if not hasattr(config.parallel_config, "moe_config"):
            config.parallel_config.moe_config = default_moe_config
        moe_config = config.parallel_config.moe_config

        self.blocks = nn.CellList()
        for i in range(config.num_layers):
            block = GPTTransformerDecoderLayer(
                hidden_size=config.hidden_size,
                batch_size=config.batch_size,
                ffn_hidden_size=config.hidden_size * config.expand_ratio,
                seq_length=config.seq_length,
                num_heads=config.num_heads,
                attention_dropout_rate=config.attention_dropout_rate,
                hidden_dropout_rate=config.hidden_dropout_rate,
                hidden_act=config.hidden_act,
                param_init_type=config.param_init_type,
                layernorm_compute_type=config.layernorm_compute_type,
                softmax_compute_type=config.softmax_compute_type,
                parallel_config=config.parallel_config.dp_mp_config,
                moe_config=moe_config)
            set_parallel_configure_for_layer(
                block, layer_id=i, layers=config.num_layers,
                offset=0, parallel_config=config.parallel_config)
            self.blocks.append(block)

        self.cast = P.Cast()
        self.tile = P.Tile().shard(((config.parallel_config.data_parallel,),))
        self.dtype = mstype.float16
        self.num_layers = config.num_layers
        self.input_position = Tensor(np.arange(config.seq_length), mstype.int32)

    def construct(self, input_ids, attention_mask):
        """GPT model"""
        batch_size, seq_length = F.shape(input_ids)
        if batch_size == 1:
            input_position = F.reshape(self.input_position, (1, seq_length))
        else:
            input_position = self.tile(self.input_position, (batch_size, 1))

        input_embedding, embedding_table = self.embedding(input_ids, input_position)

        hidden_states = self.cast(input_embedding, self.dtype)
        hidden_shape = F.shape(hidden_states)
        hidden_states = F.reshape(hidden_states, (-1, hidden_shape[-1]))

        for i in range(self.num_layers):
            hidden_states = self.blocks[i](hidden_states, attention_mask)

        output_state = self.layernorm(hidden_states)

        return output_state, embedding_table


class GPTHead(nn.Cell):
    r"""Head for GPT to get the logits of each token in the vocab."""

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 compute_type=mstype.float16,
                 parallel_config=None):
        super().__init__()
        copied_parallel_config = copy.deepcopy(parallel_config)
        mp = copied_parallel_config.model_parallel
        if vocab_size % mp != 0:
            logger.warning("The vocab size of GPTHead MatMul is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of GPTHead MatMul will be changed: mp = 1")
            copied_parallel_config.model_parallel = 1

        if copied_parallel_config.pipeline_stage > 1:
            copied_parallel_config.vocab_emb_dp = False
        if copied_parallel_config.vocab_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(((copied_parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((copied_parallel_config.data_parallel, 1), (
                copied_parallel_config.model_parallel, 1)))
        self.hidden_size = hidden_size
        self.dtype = compute_type
        self.cast = P.Cast()

    def construct(self, state, embedding_table):
        logits = self.matmul(self.cast(state, self.dtype), self.cast(embedding_table, self.dtype))
        return logits


class CrossEntropyCalculationWithMask(nn.Cell):
    """
    Cross Entropy loss
    """

    def __init__(self, is_training=None, num_labels=None):
        super(CrossEntropyCalculationWithMask, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.is_training = is_training
        self.num_labels = num_labels
        self.log_softmax = P.LogSoftmax(axis=-1)

    def construct(self, logits, label_ids, input_mask=None):
        """
        Calculate loss

        Args:
            logits (Tensor): the probability distribution over vocabulary.
            label_ids (Tensor): the indices of input sequence tokens in the vocabulary.
            input_mask (Tensor): input sentences padding mask, where 0 indicates padding position.

        Returns:
            return_value (Tensor, mstype.float32): if is_training is False, directly return the logits, otherwise,
                                                   return the computed loss.
        """

        # logits [batch * (seq_length-1), vocab_size]   label_ids [batch, seq_length-1]
        logits = self.log_softmax(logits)

        if self.is_training:
            label_ids = self.reshape(label_ids, self.last_idx)  # label_ids [batch * (seq_length-1)]
            one_hot_labels = self.onehot(label_ids, self.num_labels, self.on_value,
                                         self.off_value)  # [batch * (seq_length-1), vocab_size]
            per_example_loss = self.neg(
                self.reduce_sum(one_hot_labels * logits, self.last_idx))  # [batch * (seq_length-1)]

            # for PPL calculation in evaluation
            if input_mask is not None:
                input_mask = self.cast(self.reshape(input_mask, self.last_idx),
                                       mstype.float32)  # [batch * (seq_length-1)]

                valid_loss_sum = self.reduce_sum(input_mask * per_example_loss, ())
                valid_element_sum = self.reduce_sum(input_mask, ()) + self.cast(F.tuple_to_array((1e-5,)),
                                                                                mstype.float32)
                loss = valid_loss_sum / valid_element_sum
            else:
                loss = self.reduce_mean(per_example_loss, self.last_idx)  # a number
            return_value = self.cast(loss, mstype.float32)
        else:
            return_value = logits * 1.0  # [batch * (seq_length-1), vocab_size]

        return return_value
