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
Test Module for testing functions of Generator model class

How to run this:
linux:  pytest ./tests/st/test_model/test_t5_model/test_t5_generation.py

"""
# import pytest

from mindformers.models import T5ForConditionalGeneration, T5Tokenizer


def modify_batch_size(net, batch_size):
    if hasattr(net, 'batch_size'):
        net.batch_size = batch_size
    for cell in net.cells():
        modify_batch_size(cell, batch_size)


# @pytest.mark.level0
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.env_onecard
class TestGeneratorUseT5:
    """A test class for testing Model classes"""
    def setup_class(self):
        self.t5 = T5ForConditionalGeneration.from_pretrained("t5_small")
        self.tokenizer = T5Tokenizer.from_pretrained("t5_small")

    # @pytest.mark.parametrize('do_sample', [False, True])
    def test_batch_inference_do_sample_false(self, do_sample):
        """
        Feature: Test input as batch for generator
        Description: batch inference
        Expectation: ValueError, AttributeError
        """
        inference_words = ["hello world!", "translate the English to the Romanian: UN Chief Says There Is No Military "
                                           "Solution in Syria"]

        modify_batch_size(self.t5, batch_size=len(inference_words))
        words = self.tokenizer(inference_words, max_length=21, padding='max_length')['input_ids']
        modify_batch_size(self.t5, batch_size=2)
        output = self.t5.generate(words, do_sample=do_sample)
        output = self.tokenizer.decode(output, skip_special_tokens=True)
        assert output == ["hello world!", "eful ONU declară că nu există o soluţie militară în Siria"]


    # @pytest.mark.parametrize('do_sample', [False, True])
    def test_single_inference(self, do_sample):
        """
        Feature: Test input as single example for generator
        Description: single example inference
        Expectation: ValueError, AttributeError
        """
        words = "translate the English to the Romanian: UN Chief Says There Is No Military Solution in Syria"
        words = self.tokenizer(words, max_length=21, padding='max_length')['input_ids']
        modify_batch_size(self.t5, batch_size=1)
        output = self.t5.generate(words, do_sample=do_sample)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        assert output == "eful ONU declară că nu există o soluţie militară în Siria"


    def test_single_inference_topk(self):
        """
        Feature: Test input as single example for generator
        Description: single example inference
        Expectation: ValueError, AttributeError
        """
        words = "translate the English to the Romanian: UN Chief Says There Is No Military Solution in Syria"
        words = self.tokenizer(words, max_length=21, padding='max_length')['input_ids']
        modify_batch_size(self.t5, batch_size=1)
        output = self.t5.generate(words, do_sample=False, top_k=4)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        assert output == "eful ONU declară că nu există o soluţie militară în Siria"

    # @pytest.mark.parametrize('top_p', [0.4, 1.0, 2.0])
    def test_single_inference_topp(self, top_p):
        """
        Feature: Test input as single example for generator
        Description: single example inference
        Expectation: ValueError, AttributeError
        """
        words = "translate the English to the Romanian: UN Chief Says There Is No Military Solution in Syria"
        words = self.tokenizer(words, max_length=21, padding='max_length')['input_ids']
        modify_batch_size(self.t5, batch_size=1)
        output = self.t5.generate(words, do_sample=False, top_p=top_p)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        assert output == "eful ONU declară că nu există o soluţie militară în Siria"

    # @pytest.mark.parametrize('repetition_penalty', [0.4, 1.0, 2.0])
    def test_single_inference_repetition_penalty(self, repetition_penalty):
        """
        Feature: Test input as batch example for generator with repeation_penalty
        Description: single example inference
        Expectation: ValueError, AttributeError
        """
        inference_words = ["hello world!", "translate the English to the Romanian: UN Chief Says There Is No Military "
                                           "Solution in Syria"]
        words = self.tokenizer(inference_words, max_length=21, padding='max_length')['input_ids']
        modify_batch_size(self.t5, batch_size=2)
        output = self.t5.generate(words, do_sample=False, repetition_penalty=repetition_penalty)
        output = self.tokenizer.decode(output, skip_special_tokens=True)
        assert output

    # @pytest.mark.parametrize('max_length', [1, 32])
    def test_single_inference_max_length(self, max_length):
        """
        Feature: Test input as batch example for generator with max_length
        Description: single example inference
        Expectation: ValueError, AttributeError
        """
        inference_words = ["hello world!", "translate the English to the Romanian: UN Chief Says There Is No Military "
                                           "Solution in Syria"]
        words = self.tokenizer(inference_words, max_length=32, padding='max_length')['input_ids']
        modify_batch_size(self.t5, batch_size=2)
        output = self.t5.generate(words, do_sample=False, max_length=max_length)
        output = self.tokenizer.decode(output, skip_special_tokens=True)
        assert output
