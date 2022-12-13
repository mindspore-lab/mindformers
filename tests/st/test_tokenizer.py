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
"""Test tokenizer class"""
import os
import shutil

import pytest

from mindspore import Tensor

from mindformers import PretrainedTokenizer, AutoTokenizer
from mindformers import BertTokenizer, ClipTokenizer

class TestAutoTokenizerMethod:
    """A test class for testing the AutoTokenizer"""
    def generate_fake_vocab(self):
        vocabs = ["[PAD]", "[unused1]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "!"]
        with open(os.path.join(self.output_path, 'vocab_file.txt'), 'w') as fp:
            for item in vocabs:
                fp.write(item + '\n')

    def setup_method(self):
        self.output_path = os.path.join(os.path.dirname(__file__), 'test_tokenizer_output')
        self.bert_path_saved = os.path.join(os.path.dirname(__file__), 'test_bert_tokenizer_output')
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
        bert_tokenizer = BertTokenizer(vocab_file=os.path.join(self.output_path, 'vocab_file.txt'), do_lower_case=False,
                                       do_basic_tokenize=False)
        bert_tokenizer.save_pretrained(self.bert_path_saved)
        restore_tokenizer = AutoTokenizer.from_pretrained(self.bert_path_saved)
        assert not restore_tokenizer.init_kwargs['do_lower_case']
        assert not restore_tokenizer.init_kwargs['do_basic_tokenize']
        assert isinstance(restore_tokenizer, BertTokenizer)

    def test_save_and_load_using_clip_tokenizer(self):
        """
        Feature: The test load and save function for the tokenizer
        Description: Load the tokenizer and then saved it
        Expectation: The restored kwargs is not expected version.
        """
        clip_tokenizer = ClipTokenizer.from_pretrained("clip_vit_b_32")
        clip_tokenizer.save_pretrained(self.bert_path_saved)
        restore_tokenizer = AutoTokenizer.from_pretrained(self.bert_path_saved)
        res = clip_tokenizer.tokenize("hello world?")
        assert isinstance(restore_tokenizer, ClipTokenizer)
        assert res == ['hello</w>', 'world</w>', '?</w>']


class TestPretrainedTokenizerMethod:
    """A test class for testing the PretrainedTokenizer"""
    def generate_fake_vocab(self):
        vocabs = ["[PAD]", "[unused1]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "!"]
        with open(os.path.join(self.output_path, 'vocab.txt'), 'w') as fp:
            for item in vocabs:
                fp.write(item + '\n')

    def setup_method(self):
        self.output_path = os.path.join(os.path.dirname(__file__), 'test_tokenizer_output')
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
        tokenizer = PretrainedTokenizer.from_pretrained(self.output_path)
        with pytest.raises(NotImplementedError):
            tokenizer("hello world")


class TestBertTokenizerMethod:
    """A test class for testing the BertTokenizer"""
    def generate_fake_vocab(self):
        vocabs = ["[PAD]", "[unused1]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "!", "Hello", "World"]
        with open(os.path.join(self.output_path, 'vocab.txt'), 'w') as fp:
            for item in vocabs:
                fp.write(item + '\n')

    def setup_method(self):
        self.output_path = os.path.join(os.path.dirname(__file__), 'test_tokenizer_output')
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

    def test_call_with_one_setence_bert_tokenizer(self):
        """
        Feature: The BERT Tokenizer test using from python class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        bert_tokenizer = BertTokenizer(vocab_file=os.path.join(self.output_path, 'vocab.txt'))
        res = bert_tokenizer("hello world")
        assert res == {'attention_mask': [1, 1, 1, 1], 'input_ids': [3, 6, 7, 4],
                       'token_type_ids': [0, 0, 0, 0]}, f"The res is {res}"

    def test_call_with_two_sentence_bert_tokenizer(self):
        """
        Feature: The BERT Tokenizer test using from python class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        bert_tokenizer = BertTokenizer(vocab_file=os.path.join(self.output_path, 'vocab.txt'))
        res = bert_tokenizer("hello world", text_pair="hello world !")
        assert res == {'attention_mask': [1, 1, 1, 1, 1, 1, 1], 'input_ids': [3, 6, 7, 4, 6, 7, 8],
                       'token_type_ids': [0, 0, 0, 1, 1, 1, 1]}, f"The res is {res}"

    def test_tokenize_in_tokenizer(self):
        """
        Feature: The BERT Tokenizer test using from python class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        bert_tokenizer = BertTokenizer(vocab_file=os.path.join(self.output_path, 'vocab.txt'))
        res = bert_tokenizer.tokenize("hello world")
        assert res == ["hello", "world"], f"The res is {res} is not equal to the target"

    def test_do_lower_false_in_tokenizer(self):
        """
        Feature: The BERT Tokenizer test using from python class
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to [[6, 7]].
        """
        bert_tokenizer = BertTokenizer(vocab_file=os.path.join(self.output_path, 'vocab.txt'), do_lower_case=False)
        res = bert_tokenizer("Hello World")
        assert res == {'input_ids': [3, 9, 10, 4], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}, \
            f"The res is {res} is not equal to the target"


class TestClipTokenizerMethod:
    """Test the basic usage of the ClipTokenizer"""
    def test_padding(self):
        """
        Feature: The ClipTokenizer test using padding
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not equal to the target.
        """
        clip_tokenizer = ClipTokenizer.from_pretrained("clip_vit_b_32")
        res = clip_tokenizer("hello world?", max_length=8, padding='max_length')
        assert res == {'attention_mask': [1, 1, 1, 1, 1, 0, 0, 0],
                       'input_ids': [49406, 3306, 1002, 286, 49407, 49407, 49407, 49407]}, f"The res is {res}."

    def test_return_tensors(self):
        """
        Feature: The ClipTokenizer test using return_tensors
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not the instance of the Tensor.
        """
        clip_tokenizer = ClipTokenizer.from_pretrained("clip_vit_b_32")
        res = clip_tokenizer("hello world?", max_length=8, padding='max_length', return_tensors='ms')
        assert len(res) == 2
        for k in res.keys():
            assert isinstance(res[k], Tensor)

    def test_batch_inputs_call(self):
        """
        Feature: The ClipTokenizer test using batch_calls
        Description: Using call forward process of the tokenizer without error
        Expectation: The returned ret is not the instance of the Tensor.
        """
        clip_tokenizer = ClipTokenizer.from_pretrained("clip_vit_b_32")
        batch_inputs = ["hello world?", "Who are you?", "I am find, thank you."]
        res = clip_tokenizer(batch_inputs, max_length=12, padding='max_length')
        assert len(res) == 2
        assert res == {'input_ids': [[49406, 3306, 1002, 286, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407],
                                     [49406, 822, 631, 592, 286, 49407, 49407, 49407, 49407, 49407, 49407, 49407],
                                     [49406, 328, 687, 1416, 267, 1144, 592, 269, 49407, 49407, 49407, 49407]],
                       'attention_mask': [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]}
