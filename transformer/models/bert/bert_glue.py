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
from transformer.models.bert.utils import CrossEntropyCalculation
from transformer.processor.finetune_eval_model import BertCLSModel


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
