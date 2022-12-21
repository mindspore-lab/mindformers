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
Test Module for testing functions of T5 model class

How to run this:
linux:  pytest ./tests/ut/test_t5_model.py

"""
import os

import numpy as np
from numpy import allclose

from mindspore import Tensor

from mindformers import MindFormerBook, AutoModel
from mindformers.models import T5ModelForLoss, T5Config


class TestModelForT5Method:
    '''A test class for testing Model classes'''
    def setup_method(self):
        """get_input"""
        self.save_directory = os.path.join(MindFormerBook.get_project_path(),
                                           'checkpoint_save', 't5' + str(self))

    def test_auto_model(self):
        """
        Feature: AutoModel, from_pretrained, from_config
        Description: Test to get model instance by AutoModel.from_pretrained
                    and AutoModel.from_config
        Expectation: TypeError, ValueError, RuntimeError
        """
        AutoModel.show_support_list()

    def test_t5_model(self):
        """
        Feature: T5Model, from_pretrained, input config
        Description: Test to get model instance by ClipModel.from_pretrained
                    and input config
        Expectation: TypeError, ValueError, RuntimeError
        """

        # input model name, load model and weights
        config = T5Config(num_hidden_layers=1)
        T5ModelForLoss(config)

    def test_save_model(self):
        """
        Feature: save_pretrained method of T5Model
        Description: Test to save checkpoint for T5Model
        Expectation: ValueError, AttributeError
        """
        t5 = T5ModelForLoss(T5Config(num_hidden_layers=1, hidden_dropout_prob=0.0,
                                     attention_probs_dropout_prob=0.0,
                                     batch_size=2, seq_length=16, max_decode_length=8))
        t5.save_pretrained(self.save_directory, save_name='t5_model')
        input_ids = Tensor(np.random.randint(low=0, high=15, size=(2, 16,)).astype(np.int32))
        attention_mask = Tensor(np.random.randint(low=0, high=15, size=(2, 16,)).astype(np.int32))
        labels = Tensor(np.random.randint(low=0, high=15, size=(2, 8,)).astype(np.int32))

        out1 = t5(input_ids, attention_mask, labels)
        new_t5 = T5ModelForLoss.from_pretrained(self.save_directory)
        out2 = new_t5(input_ids, attention_mask, labels)

        assert allclose(out1.asnumpy(), out2.asnumpy())
