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
# import pytest

from numpy import allclose


from mindformers import MindFormerBook, AutoModel
from mindformers.models import T5ForConditionalGeneration, T5Config, T5Tokenizer


# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
class TestModelForT5Method:
    """A test class for testing Model classes"""
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
        assert 't5' in AutoModel.get_support_list()
        assert 't5_small' in AutoModel.get_support_list()['t5']

    def test_t5_model_with_loss(self):
        """
        Feature: T5Model, from_pretrained, input config
        Description: Test to get model instance by ClipModel.from_pretrained
                    and input config
        Expectation: TypeError, ValueError, RuntimeError
        """

        # input model name, load model and weights
        config = T5Config(num_layers=1)
        T5ForConditionalGeneration(config)

    def test_save_model(self):
        """
        Feature: save_pretrained method of T5Model using tokenization output as input
        Description: Test to save checkpoint for T5Model
        Expectation: ValueError, AttributeError
        """
        t5 = T5ForConditionalGeneration(T5Config(num_layers=1, hidden_dropout_rate=0.0, attention_dropout_rate=0.0,
                                                 batch_size=1, seq_length=16, max_decode_length=8))
        t5.save_pretrained(self.save_directory, save_name='t5_model')
        tokenizer = T5Tokenizer.from_pretrained('t5_small')

        src_output = tokenizer(["hello world"], padding='max_length', max_length=t5.config.seq_length,
                               return_tensors='ms')

        labels = tokenizer(["So happy to see you!"], padding='max_length', max_length=t5.config.max_decode_length,
                           return_tensors='ms')["input_ids"]
        input_ids = src_output['input_ids']
        attention_mask = src_output['attention_mask']

        out1 = t5(input_ids, attention_mask, labels)
        new_t5 = T5ForConditionalGeneration.from_pretrained(self.save_directory)
        out2 = new_t5(input_ids, attention_mask, labels)

        assert allclose(out1.asnumpy(), out2.asnumpy())

    def test_model_forward_loss(self):
        """
        Feature: test model forward loss of T5Model using tokenization output as input
        Description: Test to save checkpoint for T5Model
        Expectation: ValueError, AttributeError
        """
        model = T5ForConditionalGeneration.from_pretrained('t5_small', dropout_rate=0.0)
        tokenizer = T5Tokenizer.from_pretrained('t5_small')

        src_output = tokenizer(["hello world"], padding='max_length', max_length=model.config.seq_length,
                               return_tensors='ms')

        model_input = tokenizer(["So happy to see you!"], padding='max_length',
                                max_length=model.config.max_decode_length,
                                return_tensors='ms')["input_ids"]
        input_ids = src_output['input_ids']
        attention_mask = src_output['attention_mask']
        output = model(input_ids, attention_mask, model_input)
        assert output.asnumpy() < 7

        model.set_train(False)
        output = model(input_ids, attention_mask, model_input, return_loss=True)
        assert len(output.asnumpy().shape) == 1

        output = model(input_ids, attention_mask, model_input, return_loss=False)
        assert len(output.asnumpy().shape) == 2
