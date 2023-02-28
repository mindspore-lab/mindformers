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
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.transformer.transformer import default_moe_config
from mindspore.nn.transformer.layers import _LayerNorm
from mindspore.nn.transformer.transformer import AttentionMask, Transformer, VocabEmbedding
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.base_model import BaseModel
from ...mindformer_book import MindFormerBook
from .gpt2_config import Gpt2Config

__all__ = ['GPT2LMHeadModel']


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GPT2LMHeadModel(BaseModel):
    """
            Provide gpt training loss or logits through network.

            Args:
                config (Gpt2Config): The config of Gpt2Model.

            Returns:
                Tensor, the loss or logits of the network.
        """
    _support_list = MindFormerBook.get_model_support_list()['gpt2']

    def __init__(self, config: Gpt2Config = None):
        super(GPT2LMHeadModel, self).__init__(config)
        self.config = config if config is not None else Gpt2Config()
        self.is_training = (self.phase == 'train')
        self.gpt2 = GPT2LanguageModel(self.config, self.is_training)
        self.num_labels = self.config.vocab_size
        self.loss = CrossEntropyCalculationWithMask(True, num_labels=self.num_labels)
        self.add = P.Add().shard(((1,), ()))
        self.use_moe = (self.config.parallel_config.moe_config.expert_num > 1)
        self.eos_token = self.config.eos_token
        parallel_config = self.config.parallel_config
        dp = parallel_config.data_parallel
        self.stridedslice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()

    def construct(self, input_ids, input_mask, label_ids=None):
        """
        construct function for Language Modeling

        Args:
            input_ids (Tensor): the indices of input sequence tokens in the vocabulary.
            input_mask (Tensor): input sequence padding mask, where 0 indicates padding position.
            label_ids (Tensor): the indices of input sequence tokens in the vocabulary

        Returns:
            lm_logits (Tensor) or loss (mstype.float32): if is_training is False, directly return the logits,
                                                         otherwise, return the computed loss.
        """
        input_mask = P.Cast()(input_mask, mstype.float16)
        lm_logits, moe_loss = self.gpt2(input_ids, input_mask)  # [batch_size, seq_length, vocab_size]
        if not self.is_training or label_ids is None:
            return lm_logits

        shift_logits = lm_logits[::, :-1, ::]  # [batch_size, seq_length - 1, vocab_size]
        shift_logits = self.reshape(shift_logits, (-1, self.num_labels))  # [batch * (seq_length - 1), vocab_size]
        label_ids = label_ids[::, 1:]
        input_mask = input_mask[::, 1:]
        loss = self.loss(shift_logits, label_ids, input_mask)
        if not self.use_moe:
            return self.cast(loss, mstype.float32)
        total_loss = self.add(loss, moe_loss)
        return self.cast(total_loss, mstype.float32)


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
        self.get_attention_mask = AttentionMask(seq_length=config.seq_length,
                                                parallel_config=config.parallel_config.dp_mp_config)
        self.word_embedding = VocabEmbedding(vocab_size=config.vocab_size,
                                             embedding_size=config.embedding_size,
                                             param_init=initializer("truncatedNormal",
                                                                    [config.vocab_size, config.embedding_size],
                                                                    dtype=config.compute_dtype),
                                             parallel_config=config.parallel_config.embedding_dp_mp_config)

        new_parallel_config = copy.deepcopy(config.parallel_config.embedding_dp_mp_config)
        new_parallel_config.vocab_emb_dp = True

        self.position_embedding = VocabEmbedding(vocab_size=config.seq_length,
                                                 embedding_size=config.embedding_size,
                                                 param_init=initializer(TruncatedNormal(config.initializer_range),
                                                                        [config.seq_length, config.embedding_size],
                                                                        dtype=config.compute_dtype),
                                                 parallel_config=new_parallel_config)
        self.blocks = nn.CellList()
        if not hasattr(config.parallel_config, "moe_config"):
            config.parallel_config.moe_config = default_moe_config
        moe_config = config.parallel_config.moe_config
        self.transformer = Transformer(hidden_size=config.embedding_size,
                                       batch_size=config.batch_size,
                                       ffn_hidden_size=config.embedding_size * config.expand_ratio,
                                       src_seq_length=config.seq_length,
                                       tgt_seq_length=config.seq_length,
                                       encoder_layers=config.num_layers,
                                       attention_dropout_rate=config.attention_probs_dropout_prob,
                                       hidden_dropout_rate=config.hidden_dropout_prob,
                                       decoder_layers=0,
                                       param_init_type=config.compute_dtype,
                                       layernorm_compute_type=config.layernorm_dtype,
                                       softmax_compute_type=config.softmax_dtype,
                                       num_heads=config.num_heads,
                                       parallel_config=config.parallel_config,
                                       moe_config=moe_config)
        self.cast = P.Cast()
        self.dtype = config.dtype
        self.use_moe = (moe_config.expert_num > 1)
        self.layernorm = _LayerNorm((config.embedding_size,)).to_float(config.layernorm_dtype)
        self.layernorm.shard(((config.parallel_config.data_parallel, 1, 1),))
        self.add = P.Add().shard(
            ((config.parallel_config.data_parallel, 1, 1), (config.parallel_config.data_parallel, 1, 1)))

    def construct(self, input_ids, input_mask):
        """GPT model"""
        input_embedding, embedding_table = self.word_embedding(input_ids)

        batch_size, seq_length = F.shape(input_ids)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (batch_size, 1))

        position_embedding, _ = self.position_embedding(input_position)
        hidden_states = self.add(input_embedding, position_embedding)
        input_mask = P.Cast()(input_mask, self.dtype)
        attention_mask = self.get_attention_mask(input_mask)

        moe_loss = 0
        if self.use_moe:
            hidden_states, present_layer, _, moe_loss = self.transformer(hidden_states, attention_mask)
        else:
            hidden_states, present_layer, _ = self.transformer(hidden_states, attention_mask)

        output_state = self.layernorm(hidden_states)

        if self.use_moe:
            return output_state, present_layer, embedding_table, moe_loss
        return output_state, present_layer, embedding_table


class GPT2LanguageModel(nn.Cell):
    """
    GPT2LanguageModel is responsible for Language Modeling task, i.e. WikiText2, WikiText103, PTB, 1BW datasets.
    """

    def __init__(self, config, is_training):
        """
        Args:
            config: the configuration of GPT-2 model
            is_training (bool): `True` for train (finetune), `False` for evaluation.
        """
        super(GPT2LanguageModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.backbone = GPT2Model(config)
        self.vocab_size = config.vocab_size
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.dtype = config.compute_dtype
        self.dense1 = nn.Dense(config.embedding_size,
                               config.vocab_size,
                               weight_init=TruncatedNormal(0.02),
                               has_bias=False).to_float(config.compute_dtype)
        self.dropout = nn.Dropout(1 - config.dropout_prob)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.print = P.Print()
        self.use_moe = self.backbone.use_moe

    def construct(self, input_ids, input_mask):
        """
        Construct network.

        Args:
            input_ids (Tensor): input sentences with shape [batch_size, seq_len].
            input_mask (Tensor): input sentences padding mask with shape [batch_size, seq_len],
                                 where 0 indicates padding position.

        Returns:
            lm_logits (Tensor): language model distribution with log_softmax, shape with[batch_size, seq_len, d_model].
        """
        moe_loss = 0
        if self.use_moe:
            output, _, _, moe_loss = self.backbone(input_ids, input_mask)
        else:
            output, _, _ = self.backbone(input_ids, input_mask)
        output = self.cast(output, self.dtype)
        batch_size, seq_length, d_model = self.shape(output)
        output_reshape = P.Reshape()(output, (-1, d_model))  # [batch_size * seq_len, d_model]
        logits = self.dense1(output_reshape)
        logits = self.cast(logits, self.dtype)
        lm_logits = P.Reshape()(logits, (batch_size, seq_length, self.vocab_size))  # [batch_size, seq_len, vocab]
        return lm_logits, moe_loss


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
