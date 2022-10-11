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
"""
Auto Config
"""
from collections import OrderedDict
from transformer.models.bert import BertConfig, BertPreTraining, BertNetworkWithLoss
from transformer.models.gpt import GPTConfig, GPT, GPTWithLoss
from transformer.models.t5 import TransformerConfig, TransformerModel, TransformerNetworkWithLoss
from transformer.models.vit import VitConfig
from transformer.models.opt import OPTConfig, OPT, OPTWithLoss

from transformer.data.gpt_dataset import create_gpt_dataset
from transformer.data.bert_dataset import create_bert_dataset, create_squad_dataset
from transformer.data.t5_dataset import create_t5_dataset
from transformer.data.wiki_dataset import create_wiki_dataset

from transformer.models.bert.bert_squad import BertSquad


CONFIG_MAPPING = OrderedDict(
    [
        ('gpt', GPTConfig),
        ('bert', BertConfig),
        ('bert_squad', BertConfig),
        ('t5', TransformerConfig),
        ('vit', VitConfig),
        ('opt', OPTConfig),
    ]
)

NETWORK_MAPPING = OrderedDict(
    [
        ('gpt', GPT),
        ('bert', BertPreTraining),
        ('bert_squad', BertSquad),
        ('t5', TransformerModel),
        ('vit', None),
        ('opt', OPT),
    ]
)

NETWORK_WITH_LOSS_MAPPING = OrderedDict(
    [
        ('gpt', GPTWithLoss),
        ('bert', BertNetworkWithLoss),
        ('bert_squad', BertSquad),
        ('t5', TransformerNetworkWithLoss),
        ('vit', None),
        ('opt', OPTWithLoss),
    ]
)

CREATE_DATASET_MAPPING = OrderedDict(
    [
        ('gpt', create_gpt_dataset),
        ('bert', create_bert_dataset),
        ('bert_squad', create_squad_dataset),
        ('t5', create_t5_dataset),
        ('vit', None),
        ('opt', create_wiki_dataset),
    ]
)


class AutoClass:
    """
    AutoClass
    """

    @staticmethod
    def get_config_class(model_key):
        """get config class"""
        if model_key in CONFIG_MAPPING.keys():
            return CONFIG_MAPPING[model_key]
        return None

    @staticmethod
    def get_network_class(model_key):
        """get net class"""
        if model_key in NETWORK_MAPPING.keys():
            return NETWORK_MAPPING[model_key]
        return None

    @staticmethod
    def get_network_with_loss_class(model_key):
        """get net with loss class"""
        if model_key in NETWORK_WITH_LOSS_MAPPING.keys():
            return NETWORK_WITH_LOSS_MAPPING[model_key]
        return None

    @staticmethod
    def get_create_dataset_func(model_key):
        """get create dataset function"""
        if model_key in CREATE_DATASET_MAPPING.keys():
            return CREATE_DATASET_MAPPING[model_key]
        return None
