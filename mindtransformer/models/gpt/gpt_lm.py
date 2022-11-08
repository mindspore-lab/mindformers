# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""GPT-2 finetune for downstream task"""
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.initializer import TruncatedNormal
from mindtransformer.models.gpt.utils import CrossEntropyCalculationWithMask
from mindtransformer.models.gpt import GPTModel


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
            config.dropout_rate = 0.0

        self.backbone = GPTModel(config)
        self.vocab_size = config.vocab_size
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.dtype = config.compute_dtype
        self.dense1 = nn.Dense(config.hidden_size,
                               config.vocab_size,
                               weight_init=TruncatedNormal(0.02),
                               has_bias=False).to_float(config.compute_dtype)
        self.dropout = nn.Dropout(1 - config.dropout_rate)
        self.log_softmax = P.LogSoftmax(axis=-1)

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
        output, _ = self.backbone(input_ids, input_mask)
        output = self.cast(output, self.dtype)
        batch_size, seq_length, d_model = self.shape(output)
        output_reshape = P.Reshape()(output, (-1, d_model))  # [batch_size * seq_len, d_model]
        logits = self.dense1(output_reshape)
        logits = self.cast(logits, self.dtype)
        logits = self.log_softmax(logits)
        lm_logits = P.Reshape()(logits, (batch_size, seq_length, self.vocab_size))  # [batch_size, seq_len, vocab]

        return lm_logits


class GPT2LM(nn.Cell):
    """
    Train interface for Language Modeling finetuning task.

    Args:
        config (class): the configuration of GPT-2 model.
        is_training (bool): whether to train.
        use_one_hot_embeddings (bool): whether to use onehot embeddings.
    """

    def __init__(self, config=None):
        super(GPT2LM, self).__init__()
        self.gpt2 = GPT2LanguageModel(config, config.is_training)
        self.num_labels = config.vocab_size
        self.loss = CrossEntropyCalculationWithMask(True,
                                                    num_labels=self.num_labels,
                                                    config=config)
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
        lm_logits = self.gpt2(input_ids, input_mask)  # [batch_size, seq_length, vocab_size]

        if label_ids is None:
            return lm_logits

        shift_logits = lm_logits[::, :-1, ::]  # [batch_size, seq_length - 1, vocab_size]
        shift_logits = self.reshape(shift_logits, (-1, self.num_labels))  # [batch * (seq_length - 1), vocab_size]
        label_ids = label_ids[::, 1:]
        input_mask = input_mask[::, 1:]

        loss = self.loss(shift_logits, label_ids, input_mask)
        return loss
