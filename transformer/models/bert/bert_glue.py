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
from mindspore.nn import MSELoss
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from transformer.models.bert import BertModel
from transformer.models.bert.utils import CrossEntropyCalculation
from transformer.models.nezha import NezhaModel


class BertCLSModel(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config):
        super(BertCLSModel, self).__init__()
        if not config.is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        if config.model_type == 'bert':
            print("finetunine bert")
            self.bert = BertModel(config, config.is_training, config.use_one_hot_embeddings)
        elif config.model_type == 'nezha':
            print("finetunine nezha")
            self.nezha = NezhaModel(config, config.is_training, use_one_hot_embeddings=True)
        self.model_type = config.model_type
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = config.num_labels
        self.dense_1 = nn.Dense(config.embedding_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_dtype)
        self.dropout = nn.Dropout(1 - config.dropout_prob)
        self.assessment_method = config.assessment_method

    def construct(self, input_ids, input_mask, token_type_id):
        "Finetuning BERT model on classification tasks"
        if self.model_type == 'bert':
            _, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
        else:
            _, pooled_output, _ = self.nezha(input_ids, token_type_id, input_mask)
        cls = self.cast(pooled_output, self.dtype)
        cls = self.dropout(cls)
        logits = self.dense_1(cls)
        logits = self.cast(logits, self.dtype)
        if self.assessment_method != "spearman_correlation":
            logits = self.log_softmax(logits)
        return logits


class BertNERModel(nn.Cell):
    """
    This class is responsible for sequence labeling task evaluation, i.e. NER(num_labels=11).
    The returned output represents the final logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=11, use_crf=False, with_lstm=False,
                 dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertNERModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.hidden_size = config.embedding_size
        self.dense_1 = nn.Dense(self.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_dtype)
        if with_lstm:
            self.lstm_hidden_size = self.hidden_size // 2
            self.lstm = nn.LSTM(self.hidden_size, self.lstm_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.reshape = P.Reshape()
        self.shape = (-1, self.hidden_size)
        self.use_crf = use_crf
        self.with_lstm = with_lstm
        self.origin_shape = (-1, config.seq_length, self.num_labels)

    def construct(self, input_ids, input_mask, token_type_id):
        """Return the final logits as the results of log_softmax."""
        sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
        seq = self.dropout(sequence_output)
        if self.with_lstm:
            batch_size = input_ids.shape[0]
            data_type = self.dtype
            hidden_size = self.lstm_hidden_size
            h0 = P.Zeros()((2, batch_size, hidden_size), data_type)
            c0 = P.Zeros()((2, batch_size, hidden_size), data_type)
            seq, _ = self.lstm(seq, (h0, c0))
        seq = self.reshape(seq, self.shape)
        logits = self.dense_1(seq)
        logits = self.cast(logits, self.dtype)
        if self.use_crf:
            return_value = self.reshape(logits, self.origin_shape)
        else:
            return_value = self.log_softmax(logits)
        return return_value


class BertCLS(nn.Cell):
    """
    Train interface for classification finetuning task.
    """

    def __init__(self, config):
        super(BertCLS, self).__init__()
        self.bert = BertCLSModel(config)
        if config.num_labels == 1:
            self.loss = MSELoss()
        else:
            self.loss = CrossEntropyCalculation(config.is_training)
        self.num_labels = config.num_labels
        self.assessment_method = config.assessment_method
        self.is_training = config.is_training

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.assessment_method == "spearman_correlation":
            if self.is_training:
                loss = self.loss(logits, label_ids)
            else:
                loss = logits
        else:
            loss = self.loss(logits, label_ids, self.num_labels)
        return loss
