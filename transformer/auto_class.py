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
from transformer.models.vit import VitConfig, ViT, VitWithLoss
from transformer.models.opt import OPTConfig, OPT, OPTWithLoss

from transformer.data.gpt_dataset import create_gpt_dataset
from transformer.data.bert_dataset import create_bert_dataset
from transformer.data.downstream_dataset import create_squad_dataset, \
    create_classification_dataset, create_language_model_dataset
from transformer.data.t5_dataset import create_t5_dataset
from transformer.data.wiki_dataset import create_wiki_dataset
from transformer.data.image_dataset import create_image_dataset
from transformer.models.bert.bert_squad import BertSquad
from transformer.models.bert.bert_glue import BertCLS
from transformer.models.gpt.gpt_lm import GPT2LM, GPT2LanguageModel

CONFIG_MAPPING = OrderedDict(
    [
        ('gpt', GPTConfig),
        ('gpt_language_model', GPTConfig),
        ('bert', BertConfig),
        ('bert_squad', BertConfig),
        ('bert_glue', BertConfig),
        ('t5', TransformerConfig),
        ('vit', VitConfig),
        ('opt', OPTConfig),
    ]
)

NETWORK_MAPPING = OrderedDict(
    [
        ('gpt', GPT),
        ('gpt_language_model', GPT2LanguageModel),
        ('bert', BertPreTraining),
        ('bert_squad', BertSquad),
        ('bert_glue', BertCLS),
        ('t5', TransformerModel),
        ('vit', ViT),
        ('opt', OPT),
    ]
)

NETWORK_WITH_LOSS_MAPPING = OrderedDict(
    [
        ('gpt', GPTWithLoss),
        ('gpt_language_model', GPT2LM),
        ('bert', BertNetworkWithLoss),
        ('bert_squad', BertSquad),
        ('bert_glue', BertCLS),
        ('t5', TransformerNetworkWithLoss),
        ('vit', VitWithLoss),
        ('opt', OPTWithLoss),
    ]
)

CREATE_DATASET_MAPPING = OrderedDict(
    [
        ('gpt', create_gpt_dataset),
        ('gpt_language_model', create_language_model_dataset),
        ('bert', create_bert_dataset),
        ('bert_squad', create_squad_dataset),
        ('bert_glue', create_classification_dataset),
        ('t5', create_t5_dataset),
        ('vit', create_image_dataset),
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
        print("CONFIG_MAPPING.keys:", CONFIG_MAPPING.keys())
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
