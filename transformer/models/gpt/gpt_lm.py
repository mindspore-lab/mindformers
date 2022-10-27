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
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

from transformer.models.gpt.utils import CrossEntropyCalculationWithMask
from transformer.processor.lm_finetune import GPT2LanguageModel


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
