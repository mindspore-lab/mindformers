# Copyright 2022 Huawei Technologies Co., Ltd
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

'''
Bert for finetune script.
'''

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import TruncatedNormal
from mindspore import context
from mindtransformer.models.bert import BertModel
from mindtransformer.models.bert.utils import CrossEntropyCalculation


class BertSquad(nn.Cell):
    '''
    Train interface for SQuAD finetuning task.
    '''

    def __init__(self, config, num_labels=2, use_one_hot_embeddings=False):
        super(BertSquad, self).__init__()
        self.bert = BertSquadModel(config, config.is_training, num_labels, use_one_hot_embeddings)
        self.loss = CrossEntropyCalculation(config.is_training)
        self.num_labels = num_labels
        self.seq_length = config.seq_length
        self.is_training = config.is_training
        self.sum = P.ReduceSum()
        self.equal = P.Equal()
        self.argmax = P.ArgMaxWithValue(axis=1)
        self.squeeze = P.Squeeze(axis=-1)
        self.is_impossible = False

    def construct(self, input_ids, input_mask, token_type_id, start_position, end_position, unique_id, is_impossible):
        """interface for SQuAD finetuning task"""
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.is_impossible:
            print("is_impossible:", is_impossible)
        if self.is_training:
            unstacked_logits_0 = self.squeeze(logits[:, :, 0:1])
            unstacked_logits_1 = self.squeeze(logits[:, :, 1:2])
            start_loss = self.loss(unstacked_logits_0, start_position, self.seq_length)
            end_loss = self.loss(unstacked_logits_1, end_position, self.seq_length)
            total_loss = (start_loss + end_loss) / 2.0
        else:
            start_logits = self.squeeze(logits[:, :, 0:1])
            start_logits = start_logits + 100 * input_mask
            end_logits = self.squeeze(logits[:, :, 1:2])
            end_logits = end_logits + 100 * input_mask
            total_loss = (unique_id, start_logits, end_logits)
        return total_loss


class BertSquadModel(nn.Cell):
    '''
    This class is responsible for SQuAD
    '''

    def __init__(self, config, is_training, num_labels=2, use_one_hot_embeddings=False):
        super(BertSquadModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.hidden_size = config.embedding_size
        self.dense1 = nn.Dense(self.hidden_size, num_labels, weight_init=self.weight_init,
                               has_bias=True).to_float(config.compute_dtype)
        self.num_labels = num_labels
        self.dtype = config.dtype
        self.log_softmax = P.LogSoftmax(axis=1)
        self.is_training = is_training
        self.gpu_target = context.get_context("device_target") == "GPU"
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.shape = (-1, self.hidden_size)
        self.origin_shape = (-1, config.seq_length, self.num_labels)
        self.transpose_shape = (-1, self.num_labels, config.seq_length)

    def construct(self, input_ids, input_mask, token_type_id):
        """Return the final logits as the results of log_softmax."""
        sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
        sequence = self.reshape(sequence_output, self.shape)
        logits = self.dense1(sequence)
        logits = self.cast(logits, self.dtype)
        logits = self.reshape(logits, self.origin_shape)
        if self.gpu_target:
            logits = self.transpose(logits, (0, 2, 1))
            logits = self.log_softmax(self.reshape(logits, (-1, self.transpose_shape[-1])))
            logits = self.transpose(self.reshape(logits, self.transpose_shape), (0, 2, 1))
        else:
            logits = self.log_softmax(logits)
        return logits
