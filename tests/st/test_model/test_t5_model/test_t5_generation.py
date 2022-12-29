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
linux:  pytest ./tests/st/test_generation.py

"""
import os
import pytest

from mindformers.models import T5ModelForGeneration, T5Config, T5Tokenizer


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestModelForT5Method:
    """A test class for testing Model classes"""
    @pytest.mark.parametrize('do_sample', [True, False])
    def test_t5_generation(self, do_sample):
        """
        Feature: generator method of T5Model
        Description: Test to save checkpoint for T5Model
        Use the following commands to train the example spiece.model
            import sentencepiece as spm
            spm.SentencePieceTrainer.train(input='./mindformers/models/t5/t5.py',
                                           model_prefix='spiece',
                                           vocab_size=100,
                                           user_defined_symbols=['UNK', 'bar'])
        Expectation: ValueError, AttributeError
        """
        tokenizer = T5Tokenizer.from_pretrained(os.path.dirname(__file__))
        t5 = T5ModelForGeneration(T5Config(num_hidden_layers=1, hidden_dropout_prob=0.0,
                                           attention_probs_dropout_prob=0.0,
                                           hidden_size=512,
                                           num_heads=8,
                                           vocab_size=tokenizer.vocab_size,
                                           batch_size=1, seq_length=32,
                                           max_decode_length=8))

        words = tokenizer("class T5Model")['input_ids']
        output = t5.generate(words, do_sample=do_sample)
        tokenizer.decode(output, skip_special_tokens=True)

def test_t5_model_generation_from_url():
    """
    Feature: Test generation with remote url and tokenizer
    Description: Test the generator
    Expectation: ValueError, AttributeError
    """
    t5 = T5ModelForGeneration.from_pretrained("t5_small")
    tokenizer = T5Tokenizer.from_pretrained("t5_small")
    words = tokenizer("translate the English to the Romanian: UN Chief Says There Is No Military "
                      "Solution in Syria")['input_ids']
    output = t5.generate(words, do_sample=False)
    output = tokenizer.decode(output, skip_special_tokens=True)
    assert output == "eful ONU declară că nu există o soluţie militară în Siri"
