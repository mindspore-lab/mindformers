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
Test Module for testing functions of Bert model class

How to run this:
linux:  pytest ./tests/ut/test_bert_auto.py

"""
import os

from mindformers import MindFormerBook, AutoModel
from mindformers.models import BertForPreTraining, BertConfig


class TestModelForBertMethod:
    '''A test class for testing Model classes'''
    def setup_method(self):
        """get_input"""
        self.save_directory = os.path.join(MindFormerBook.get_project_path(),
                                           'checkpoint_save', 'bert')

    def test_auto_model(self):
        """
        Feature: AutoModel, from_pretrained, from_config
        Description: Test to get model instance by AutoModel.from_pretrained
                    and AutoModel.from_config
        Expectation: TypeError, ValueError, RuntimeError
        """
        AutoModel.show_support_list()

    def test_bert_model(self):
        """
        Feature: BertForPreTraining, from_pretrained, input config
        Description: Test to get model instance by ClipModel.from_pretrained
                    and input config
        Expectation: TypeError, ValueError, RuntimeError
        """

        # input model name, load model and weights
        config = BertConfig(num_hidden_layers=1)
        BertForPreTraining(config)

    def test_save_model(self):
        """
        Feature: save_pretrained method of bert
        Description: Test to save checkpoint for bert
        Expectation: ValueError, AttributeError
        """
        bert = BertForPreTraining(BertConfig(num_hidden_layers=1, hidden_dropout_prob=0.0,
                                             attention_probs_dropout_prob=0.0,
                                             batch_size=2, seq_length=16))
        bert.save_pretrained(self.save_directory, save_name='bert_test')
        new_bert = BertForPreTraining.from_pretrained(self.save_directory)
        new_bert.save_pretrained(self.save_directory, save_name='bert_test')
