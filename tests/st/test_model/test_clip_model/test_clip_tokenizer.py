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
Test Module for testing Tokenizer class

How to run this:
linux:  pytest ./tests/st/test_model/test_clip_model/test_clip_tokenizer.py
"""
import os
import shutil
import time

import pytest
from mindspore import Tensor

from mindformers import Tokenizer, AutoTokenizer
from mindformers import BertTokenizer, CLIPTokenizer

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestAutoTokenizerMethod:
    """A test class for testing the AutoTokenizer"""
    def generate_fake_vocab(self):
        vocabs = ["[PAD]", "[unused1]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "!"]
        with open(os.path.join(self.output_path, 'vocab_file.txt'), 'w') as fp:
            for item in vocabs:
                fp.write(item + '\n')

    def setup_method(self):
        self.output_path = os.path.join(os.path.dirname(__file__), 'test_tokenizer_output' + str(self))
        self.bert_path_saved = os.path.join(os.path.dirname(__file__), 'test_bert_tokenizer_output' + str(self))
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.bert_path_saved, exist_ok=True)
        self.generate_fake_vocab()

    def teardown_method(self):
        shutil.rmtree(self.output_path)
        shutil.rmtree(self.bert_path_saved)

    def test_save_and_load_using_bert_tokenizer(self):
        """
        Feature: The test load and save function for the clip tokenizer
        Description: Load the tokenizer and then saved it
        Expectation: The restored kwargs is not expected version.
        """
        time.sleep(5)
        bert_tokenizer = BertTokenizer(vocab_file=os.path.join(self.output_path, 'vocab_file.txt'), do_lower_case=False,
                                       do_basic_tokenize=False)
        bert_tokenizer.save_pretrained(self.bert_path_saved)
        restore_tokenizer = AutoTokenizer.from_pretrained(self.bert_path_saved)
        assert not restore_tokenizer.init_kwargs['do_lower_case']
        assert not restore_tokenizer.init_kwargs['do_basic_tokenize']
        assert isinstance(restore_tokenizer, BertTokenizer)

        clip_tokenizer = CLIPTokenizer.from_pretrained("clip_vit_b_32")
        clip_tokenizer.save_pretrained(self.bert_path_saved)
        restore_tokenizer = AutoTokenizer.from_pretrained(self.bert_path_saved)
        res = clip_tokenizer.tokenize("hello world?")
        assert isinstance(restore_tokenizer, CLIPTokenizer)
        assert res == ['hello</w>', 'world</w>', '?</w>']


    def test_load_from_yaml(self):
        """
        Feature: The test load from yaml and save as the yaml for the tokenizer
        Description: Load the tokenizer and then saved it
        Expectation: The restored kwargs is not expected version.
        """
        tokenizer = AutoTokenizer.from_pretrained("clip_vit_b_32")
        res = tokenizer.tokenize("hello world?")
        assert isinstance(tokenizer, CLIPTokenizer)
        assert res == ['hello</w>', 'world</w>', '?</w>']

    def test_save_from_yaml(self):
        """
        Feature: The test save to yaml files for the tokenizer
        Description: Load the tokenizer and then saved it
        Expectation: The restored kwargs is not expected version.
        """
        bert_tokenizer = BertTokenizer(vocab_file=os.path.join(self.output_path, 'vocab_file.txt'), do_lower_case=False,
                                       do_basic_tokenize=False)
        bert_tokenizer.save_pretrained(self.bert_path_saved, file_format='yaml')
        tokenizer = AutoTokenizer.from_pretrained(self.bert_path_saved)
        assert isinstance(tokenizer, BertTokenizer)
        assert not tokenizer.do_basic_tokenize
        assert not tokenizer.do_lower_case
        res = tokenizer.tokenize("hello world!")
        assert res == ['hello', 'world', '[UNK]']

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestPretrainedTokenizerMethod:
    """A test class for testing the PretrainedTokenizer"""
    def generate_fake_vocab(self):
        vocabs = ["[PAD]", "[unused1]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "!"]
        with open(os.path.join(self.output_path, 'vocab.txt'), 'w') as fp:
            for item in vocabs:
                fp.write(item + '\n')

    def setup_method(self):
        self.output_path = os.path.join(os.path.dirname(__file__), 'test_tokenizer_output' + str(self))
        os.makedirs(self.output_path, exist_ok=True)
        self.generate_fake_vocab()

    def teardown_method(self):
        shutil.rmtree(self.output_path)

    def test_from_pretrained_tokenizer(self):
        """
        Feature: The Tokenizer test using from python class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        time.sleep(10)
        tokenizer = Tokenizer.from_pretrained(self.output_path)
        with pytest.raises(NotImplementedError):
            tokenizer("hello world")

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestBertTokenizerMethod:
    """A test class for testing the BertTokenizer"""
    def generate_fake_vocab(self):
        vocabs = ["[PAD]", "[unused1]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "!", "Hello", "World"]
        with open(os.path.join(self.output_path, 'vocab.txt'), 'w') as fp:
            for item in vocabs:
                fp.write(item + '\n')

    def setup_method(self):
        self.output_path = os.path.join(os.path.dirname(__file__), 'test_tokenizer_output' + str(self))
        os.makedirs(self.output_path, exist_ok=True)
        self.generate_fake_vocab()

    def teardown_method(self):
        shutil.rmtree(self.output_path)

    def test_from_pretrained_tokenizer(self):
        """
        Feature: The BertTokenizer test using from python class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        bert_tokenizer = BertTokenizer.from_pretrained(self.output_path)
        res = bert_tokenizer("hello world")
        assert res == {'attention_mask': [1, 1, 1, 1], 'input_ids': [3, 6, 7, 4],
                       'token_type_ids': [0, 0, 0, 0]}, f"The res is {res}"

        bert_tokenizer = BertTokenizer(
            vocab_file=os.path.join(self.output_path, 'vocab.txt'))
        res = bert_tokenizer("hello world")
        assert res == {'attention_mask': [1, 1, 1, 1], 'input_ids': [3, 6, 7, 4],
                       'token_type_ids': [0, 0, 0, 0]}, f"The res is {res}"

        bert_tokenizer = BertTokenizer(
            vocab_file=os.path.join(self.output_path, 'vocab.txt'))
        res = bert_tokenizer("hello world", text_pair="hello world !")
        assert res == {'attention_mask': [1, 1, 1, 1, 1, 1, 1],
                       'input_ids': [3, 6, 7, 4, 6, 7, 8],
                       'token_type_ids': [0, 0, 0, 1, 1, 1, 1]}, f"The res is {res}"

        bert_tokenizer = BertTokenizer(
            vocab_file=os.path.join(self.output_path, 'vocab.txt'))
        res = bert_tokenizer.tokenize("hello world")
        assert res == ["hello", "world"], f"The res is {res} is not equal to the target"

        bert_tokenizer = BertTokenizer(
            vocab_file=os.path.join(self.output_path, 'vocab.txt'), do_lower_case=False)
        res = bert_tokenizer("Hello World")
        assert res == {'input_ids': [3, 9, 10, 4],
                       'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}, \
            f"The res is {res} is not equal to the target"

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestClipTokenizerMethod:
    """Test the basic usage of the CLIPTokenizer"""
    def test_padding(self):
        """
        Feature: The CLIPTokenizer test using padding
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to the target.
        """
        clip_tokenizer = CLIPTokenizer.from_pretrained("clip_vit_b_32")
        res = clip_tokenizer("hello world?", max_length=8, padding='max_length')
        pad_id = clip_tokenizer.pad_token_id
        assert res == {'attention_mask': [1, 1, 1, 1, 1, 0, 0, 0],
                       'input_ids': [49406, 3306, 1002, 286, 49407, pad_id, pad_id, pad_id]}, f"The res is {res}."

        clip_tokenizer = CLIPTokenizer.from_pretrained("clip_vit_b_32")
        res = clip_tokenizer("hello world?", max_length=8, padding='max_length', return_tensors='ms')
        assert len(res) == 2
        for k in res.keys():
            assert isinstance(res[k], Tensor)

        clip_tokenizer = CLIPTokenizer.from_pretrained("clip_vit_b_32")
        batch_inputs = ["hello world?", "Who are you?", "I am find, thank you."]
        res = clip_tokenizer(batch_inputs, max_length=12, padding='max_length')
        assert len(res) == 2
        assert res == {'input_ids': [[49406, 3306, 1002, 286, 49407, pad_id, pad_id, pad_id, pad_id,
                                      pad_id, pad_id, pad_id],
                                     [49406, 822, 631, 592, 286, 49407, pad_id, pad_id, pad_id,
                                      pad_id, pad_id, pad_id],
                                     [49406, 328, 687, 1416, 267, 1144, 592, 269, 49407, pad_id, pad_id, pad_id]],
                       'attention_mask': [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]}
