# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test tokenizer base."""
import os
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch
from collections import OrderedDict

import mindspore as ms
import numpy as np
import pytest
import yaml

from tests.st.test_ut.test_tokenizers.get_vocab_model import get_sp_vocab_model
from mindformers import AutoTokenizer, LlamaTokenizer
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.tokenization_utils import PaddingStrategy, TruncationStrategy
from mindformers.models.tokenization_utils_base import (
    AddedToken, BatchEncoding, CharSpan, is_experimental_mode, PreTrainedTokenizerBase, SpecialTokensMixin, TensorType,
    to_py_obj, TokenSpan
)

value = OrderedDict([
    ('ChatGLM4Tokenizer', 'GLMProcessor'),
    ('LlamaTokenizer', 'LlamaProcessor')
    ])


# pylint: disable=W0212
class TestTokenizerBase(unittest.TestCase):
    """ A test class for testing base tokenizer."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())
        cls.path = cls.temp_dir
        cls.string = "An increasing sequence: one, two, three."
        get_sp_vocab_model("llama2_7b", cls.path)
        cls.tokenizer_model_path = os.path.join(cls.path, "llama2_7b_tokenizer.model")
        cls.tokenizer = LlamaTokenizer(cls.tokenizer_model_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_origin_pretrained(self):
        """test from origin pretrained."""
        with patch.object(MindFormerBook, "_TOKENIZER_NAME_TO_PROCESSOR", value):
            create_yaml("llama2_7b", self.path)
            real_tokenizer_model_path = os.path.join(self.path, "tokenizer.model")
            if os.path.exists(self.tokenizer_model_path):
                os.rename(self.tokenizer_model_path, real_tokenizer_model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.path)
            tokenizer.save_pretrained(self.path)
            tokenizer.save_pretrained(self.path, save_json=True)
            res = tokenizer.encode(self.string)
            assert res == [1, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150]
            res = self.tokenizer.encode(self.string)
            assert res == [1, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_special_tokens(self):
        """test add special tokens."""
        special_tokens_dict = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "sep_token": "<sep>",
            "pad_token": "<pad>",
            "cls_token": "<cls>",
            "mask_token": "<mask>",
            "additional_special_tokens": ["<additional1>", "<additional2>"]
        }
        res = self.tokenizer.add_special_tokens(special_tokens_dict)
        assert res == 6

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_origin_pretrained(self):
        """test save origin pretrained."""
        with patch.object(MindFormerBook, "_TOKENIZER_NAME_TO_PROCESSOR", value):
            self.tokenizer.save_origin_pretrained("notexist")
            assert os.path.exists(os.path.join("notexist", "mindspore_model.yaml"))
            self.tokenizer.save_origin_pretrained(self.path, None)
            assert os.path.exists(os.path.join(self.path, "mindspore_model.yaml"))
            with pytest.raises(ValueError):
                self.tokenizer.save_origin_pretrained(self.path, None, 'txt')

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_padding_truncation_strategies(self):
        """test get padding truncation strategies."""
        res = self.tokenizer._get_padding_truncation_strategies(padding=True)
        assert res == (PaddingStrategy.LONGEST, TruncationStrategy.DO_NOT_TRUNCATE, None, {})
        res = self.tokenizer._get_padding_truncation_strategies(padding=True, max_length=2048)
        assert res == (PaddingStrategy.LONGEST, TruncationStrategy.DO_NOT_TRUNCATE, 2048, {})
        res = self.tokenizer(self.string, padding=PaddingStrategy.LONGEST)
        assert res['input_ids'] == [1, 48, 87, 85, 157, 65, 135, 67, 135, 80, 150]
        kwargs = {'truncation_strategy': 'only_first'}
        res = (PaddingStrategy.LONGEST, None, 2048, kwargs)
        assert res == (PaddingStrategy.LONGEST, None, 2048, {'truncation_strategy': 'only_first'})

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_apply_chat_template(self):
        conversation = "Conversation"
        res = self.tokenizer.apply_chat_template(self, conversation, None, padding=True)
        assert res == [134, 0, 105, 23, 143, 140, 159, 144, 139, 105]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_truncate_sequences(self):
        """test truncate sequences."""
        ids = [1, 2, 3, 4, 5]
        pair_ids = [1, 2, 3]
        res = self.tokenizer.truncate_sequences(ids, None, 1)
        assert res == ([1, 2, 3, 4], None, [5])
        res = self.tokenizer.truncate_sequences(ids, None, 6, TruncationStrategy.ONLY_FIRST)
        assert res == ([1, 2, 3, 4, 5], None, [])
        res = self.tokenizer.truncate_sequences(ids, None, 6, TruncationStrategy.LONGEST_FIRST)
        assert res == ([1, 2, 3, 4, 5], None, [])
        res = self.tokenizer.truncate_sequences(ids, pair_ids, 6, TruncationStrategy.LONGEST_FIRST)
        assert res == ([1], [1], [])
        res = self.tokenizer.truncate_sequences(ids, pair_ids, 1, TruncationStrategy.ONLY_SECOND)
        assert res == ([1, 2, 3, 4, 5], [1, 2], [3])

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_encode_decode(self):
        """test encode and decode."""
        text = "Hello world"
        encoded = self.tokenizer.encode(text)
        assert isinstance(encoded, list)
        decoded = self.tokenizer.decode(encoded)
        assert isinstance(decoded, str)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_batch_encode_plus(self):
        """test batch encode plus."""
        texts = ["Hello world", "This is a test"]
        res = self.tokenizer.batch_encode_plus(texts, padding=True)
        assert 'input_ids' in res
        assert 'attention_mask' in res
        assert len(res['input_ids']) == 2

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_batch_decode(self):
        """test batch decode."""
        sequences = [[1, 48, 87, 85], [1, 48, 87]]
        res = self.tokenizer.batch_decode(sequences)
        assert isinstance(res, list)
        assert len(res) == 2

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pad(self):
        """test pad method."""
        encoded_inputs = {
            'input_ids': [[1, 2, 3], [1, 2, 3, 4, 5]],
        }
        res = self.tokenizer.pad(encoded_inputs, padding=True)
        assert 'input_ids' in res
        assert 'attention_mask' in res
        assert len(res['input_ids'][0]) == len(res['input_ids'][1])

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_special_tokens_properties(self):
        """test special tokens properties."""
        assert self.tokenizer.bos_token is not None
        assert self.tokenizer.eos_token is not None
        assert self.tokenizer.unk_token is not None
        assert self.tokenizer.pad_token is not None
        assert isinstance(self.tokenizer.bos_token_id, int)
        assert isinstance(self.tokenizer.eos_token_id, int)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_special_tokens_map(self):
        """test special tokens map."""
        special_tokens = self.tokenizer.special_tokens_map
        assert isinstance(special_tokens, dict)
        assert 'bos_token' in special_tokens or 'eos_token' in special_tokens

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_all_special_tokens(self):
        """test all special tokens."""
        all_tokens = self.tokenizer.all_special_tokens
        assert isinstance(all_tokens, list)
        assert len(all_tokens) > 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_all_special_ids(self):
        """test all special ids."""
        all_ids = self.tokenizer.all_special_ids
        assert isinstance(all_ids, list)
        assert len(all_ids) > 0
        assert all(isinstance(x, int) for x in all_ids)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_encode_plus(self):
        """test encode plus."""
        text = "Hello world"
        res = self.tokenizer.encode_plus(text, padding=True, truncation=True, max_length=10)
        assert 'input_ids' in res
        assert isinstance(res['input_ids'], list)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_tokenizer_call(self):
        """test tokenizer call method."""
        text = "Hello world"
        res = self.tokenizer(text, padding=True, truncation=True, max_length=10)
        assert 'input_ids' in res
        assert isinstance(res['input_ids'], list)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_prepare_for_model(self):
        """test prepare for model."""
        ids = [1, 2, 3, 4, 5]
        res = self.tokenizer.prepare_for_model(ids, padding=True, max_length=10)
        assert 'input_ids' in res
        assert isinstance(res['input_ids'], list)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_inputs_with_special_tokens(self):
        """test build inputs with special tokens."""
        ids = [1, 2, 3]
        res = self.tokenizer.build_inputs_with_special_tokens(ids)
        assert isinstance(res, list)
        assert len(res) >= len(ids)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_create_token_type_ids_from_sequences(self):
        """test create token type ids from sequences."""
        ids = [1, 2, 3]
        pair_ids = [4, 5]
        res = self.tokenizer.create_token_type_ids_from_sequences(ids)
        add_length = 1 if self.tokenizer.add_bos_token else 0
        add_length += 1 if self.tokenizer.add_eos_token else 0
        real_ids_length = add_length + len(ids)
        real_pair_ids_length = add_length + len(pair_ids)
        assert res == [0] * real_ids_length
        res = self.tokenizer.create_token_type_ids_from_sequences(ids, pair_ids)
        assert res == [0] * real_ids_length + [1] * real_pair_ids_length

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_vocab(self):
        """test get vocab."""
        vocab = self.tokenizer.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) > 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_vocab_size(self):
        """test vocab size."""
        vocab_size = self.tokenizer.vocab_size
        assert isinstance(vocab_size, int)
        assert vocab_size > 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_model_max_length(self):
        """test model max length."""
        max_length = self.tokenizer.model_max_length
        assert isinstance(max_length, int)
        assert max_length > 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_padding_side(self):
        """test padding side."""
        padding_side = self.tokenizer.padding_side
        assert padding_side in ['left', 'right']

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_truncation_side(self):
        """test truncation side."""
        truncation_side = self.tokenizer.truncation_side
        assert truncation_side in ['left', 'right']

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_clean_up_tokenization(self):
        """test clean up tokenization."""
        text = "This is a test . It works !"
        cleaned = self.tokenizer.clean_up_tokenization(text)
        assert isinstance(cleaned, str)
        assert cleaned == "This is a test. It works!"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_special_tokens_mask(self):
        """test get special tokens mask."""
        token_ids = [1, 2, 3, 4, 5]
        mask = self.tokenizer.get_special_tokens_mask(token_ids, already_has_special_tokens=True)
        assert isinstance(mask, list)
        assert len(mask) == len(token_ids)
        assert all(x in [0, 1] for x in mask)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_tokens_to_string(self):
        """test convert tokens to string."""
        tokens = self.tokenizer.tokenize(self.string)
        if tokens:
            result = self.tokenizer.convert_tokens_to_string(tokens)
            assert isinstance(result, str)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_num_special_tokens_to_add(self):
        """test num special tokens to add."""
        num_single = self.tokenizer.num_special_tokens_to_add(pair=False)
        num_pair = self.tokenizer.num_special_tokens_to_add(pair=True)
        assert isinstance(num_single, int)
        assert isinstance(num_pair, int)
        assert num_pair >= num_single

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_tokenize(self):
        """test tokenize method."""
        text = "Hello world"
        tokens = self.tokenizer.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_is_fast(self):
        """test is_fast property."""
        is_fast = self.tokenizer.is_fast
        assert isinstance(is_fast, bool)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_max_len_single_sentence(self):
        """test max len single sentence."""
        max_len = self.tokenizer.max_len_single_sentence
        assert isinstance(max_len, int)
        assert max_len > 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_max_len_sentences_pair(self):
        """test max len sentences pair."""
        max_len = self.tokenizer.max_len_sentences_pair
        assert isinstance(max_len, int)
        assert max_len > 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pad_with_different_strategies(self):
        """test pad with different padding strategies."""
        encoded_inputs = {
            'input_ids': [[1, 2, 3], [1, 2, 3, 4, 5]],
        }
        # Test longest padding
        res = self.tokenizer.pad(encoded_inputs, padding='longest')
        assert len(res['input_ids'][0]) == len(res['input_ids'][1])
        # Test max_length padding
        res = self.tokenizer.pad(encoded_inputs, padding='max_length', max_length=10)
        assert len(res['input_ids'][0]) == 10

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_truncation_with_stride(self):
        """test truncation with stride."""
        ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        res = self.tokenizer.truncate_sequences(ids, None, 5, stride=2)
        assert len(res[2]) > 0  # overflowing tokens should exist

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_batch_encode_with_pairs(self):
        """test batch encode with text pairs."""
        texts = ["Hello", "World"]
        text_pairs = ["there", "here"]
        res = self.tokenizer.batch_encode_plus(
            list(zip(texts, text_pairs)),
            padding=True
        )
        assert 'input_ids' in res
        assert len(res['input_ids']) == 2

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_prepare_for_model_with_pairs(self):
        """test prepare for model with pair ids."""
        ids = [1, 2, 3]
        pair_ids = [4, 5, 6]
        res = self.tokenizer.prepare_for_model(
            ids,
            pair_ids=pair_ids,
            add_special_tokens=True,
            padding='max_length',
            max_length=15
        )
        assert 'input_ids' in res
        assert len(res['input_ids']) == 15

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_decode_skip_special_tokens(self):
        """test decode with skip special tokens."""
        encoded = self.tokenizer.encode(self.string)
        decoded_with_special = self.tokenizer.decode(encoded, skip_special_tokens=False)
        decoded_without_special = self.tokenizer.decode(encoded, skip_special_tokens=True)
        assert isinstance(decoded_with_special, str)
        assert isinstance(decoded_without_special, str)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_added_tokens_decoder(self):
        """test added tokens decoder."""
        added_tokens = self.tokenizer.added_tokens_decoder
        assert isinstance(added_tokens, dict)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_tokenizer_len(self):
        """test tokenizer length."""
        length = len(self.tokenizer)
        assert isinstance(length, int)
        assert length > 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_tokenizer_repr(self):
        """test tokenizer representation."""
        repr_str = repr(self.tokenizer)
        assert isinstance(repr_str, str)
        assert 'LlamaTokenizer' in repr_str

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_encode_empty_string(self):
        """test encode empty string."""
        res = self.tokenizer.encode("")
        assert isinstance(res, list)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pad_empty_inputs(self):
        """test pad with empty inputs."""
        encoded_inputs = {'input_ids': []}
        res = self.tokenizer.pad(encoded_inputs, padding=True)
        assert 'input_ids' in res

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_truncation_with_do_not_truncate(self):
        """test truncation with do not truncate strategy."""
        ids = [1, 2, 3, 4, 5]
        res = self.tokenizer.truncate_sequences(ids, None, 0, TruncationStrategy.DO_NOT_TRUNCATE)
        assert res == (ids, None, [])

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_prepare_for_model_return_special_tokens_mask(self):
        """test prepare for model return special tokens mask."""
        ids = [1, 2, 3]
        res = self.tokenizer.prepare_for_model(
            ids,
            add_special_tokens=True,
            return_special_tokens_mask=True
        )
        assert 'special_tokens_mask' in res
        assert isinstance(res['special_tokens_mask'], list)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_prepare_for_model_return_length(self):
        """test prepare for model return length."""
        ids = [1, 2, 3]
        res = self.tokenizer.prepare_for_model(
            ids,
            return_length=True
        )
        assert 'length' in res
        assert isinstance(res['length'], int)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_special_token_setters(self):
        """test special token setters."""
        original_pad = self.tokenizer.pad_token
        original_eos = self.tokenizer.eos_token
        # Test setting and getting
        self.tokenizer.pad_token = original_pad
        assert self.tokenizer.pad_token == original_pad
        self.tokenizer.eos_token = original_eos
        assert self.tokenizer.eos_token == original_eos

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_model_input_names(self):
        """test model input names."""
        input_names = self.tokenizer.model_input_names
        assert isinstance(input_names, list)
        assert 'input_ids' in input_names

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestBatchEncoding(unittest.TestCase):
    """Test BatchEncoding class from tokenization_utils_base."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_batch_encoding_init(self):
        """test batch encoding initialization."""
        data = {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}
        encoding = BatchEncoding(data)
        assert encoding['input_ids'] == [1, 2, 3]
        assert encoding['attention_mask'] == [1, 1, 1]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_batch_encoding_getitem(self):
        """test batch encoding getitem."""
        data = {'input_ids': [[1, 2], [3, 4]], 'attention_mask': [[1, 1], [1, 1]]}
        encoding = BatchEncoding(data)
        # Test string key
        assert encoding['input_ids'] == [[1, 2], [3, 4]]
        # Test slice
        sliced = encoding[0:1]
        assert 'input_ids' in sliced

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_batch_encoding_keys_values_items(self):
        """test batch encoding keys, values, items."""
        data = {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}
        encoding = BatchEncoding(data)
        assert 'input_ids' in encoding.keys()
        assert [1, 2, 3] in list(encoding.values())
        assert ('input_ids', [1, 2, 3]) in list(encoding.items())

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_batch_encoding_convert_to_tensors_ms(self):
        """test batch encoding convert to mindspore tensors."""
        data = {'input_ids': [1, 2, 3]}
        encoding = BatchEncoding(data, tensor_type=TensorType.MINDSPORE)
        assert isinstance(encoding['input_ids'], ms.Tensor)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_batch_encoding_convert_to_tensors_np(self):
        """test batch encoding convert to numpy arrays."""
        data = {'input_ids': [1, 2, 3]}
        encoding = BatchEncoding(data, tensor_type=TensorType.NUMPY)
        assert isinstance(encoding['input_ids'], np.ndarray)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_batch_encoding_is_fast(self):
        """test batch encoding is_fast property."""
        data = {'input_ids': [1, 2, 3]}
        encoding = BatchEncoding(data)
        assert isinstance(encoding.is_fast, bool)
        assert not encoding.is_fast  # No encodings provided

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_batch_encoding_n_sequences(self):
        """test batch encoding n_sequences property."""
        data = {'input_ids': [1, 2, 3]}
        encoding = BatchEncoding(data, n_sequences=1)
        assert encoding.n_sequences == 1


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions from tokenization_utils_base."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_to_py_obj_dict(self):
        """test to_py_obj with dict input."""
        obj = {'a': ms.Tensor([1, 2, 3]), 'b': [4, 5]}
        result = to_py_obj(obj)
        assert isinstance(result, dict)
        assert isinstance(result['a'], list)
        assert result['a'] == [1, 2, 3]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_to_py_obj_list(self):
        """test to_py_obj with list input."""
        obj = [ms.Tensor([1, 2]), ms.Tensor([3, 4])]
        result = to_py_obj(obj)
        assert isinstance(result, list)
        assert result == [[1, 2], [3, 4]]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_to_py_obj_tensor(self):
        """test to_py_obj with mindspore tensor."""
        obj = ms.Tensor([1, 2, 3])
        result = to_py_obj(obj)
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_to_py_obj_numpy(self):
        """test to_py_obj with numpy array."""
        obj = np.array([1, 2, 3])
        result = to_py_obj(obj)
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_to_py_obj_primitive(self):
        """test to_py_obj with primitive types."""
        assert to_py_obj(5) == 5
        assert to_py_obj("hello") == "hello"
        assert to_py_obj([1, 2, 3]) == [1, 2, 3]


class TestAddedToken(unittest.TestCase):
    """Test AddedToken class from tokenization_utils_base."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_added_token_init(self):
        """test added token initialization."""
        token = AddedToken("<pad>", special=True)
        assert token.content == "<pad>"
        assert token.special is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_added_token_str(self):
        """test added token string representation."""
        token = AddedToken("<pad>")
        assert str(token) == "<pad>"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_added_token_normalized_default(self):
        """test added token normalized default value."""
        token_special = AddedToken("<pad>", special=True)
        token_normal = AddedToken("hello", special=False)
        assert token_special.normalized is False
        assert token_normal.normalized is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_added_token_options(self):
        """test added token with various options."""
        token = AddedToken(
            "<test>",
            single_word=True,
            lstrip=True,
            rstrip=True,
            special=True,
            normalized=False
        )
        assert token.single_word is True
        assert token.lstrip is True
        assert token.rstrip is True
        assert token.special is True
        assert token.normalized is False


class TestNamedTuples(unittest.TestCase):
    """Test NamedTuples from tokenization_utils_base."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_char_span(self):
        """test CharSpan named tuple."""
        span = CharSpan(start=0, end=5)
        assert span.start == 0
        assert span.end == 5
        assert isinstance(span, tuple)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_token_span(self):
        """test TokenSpan named tuple."""
        span = TokenSpan(start=0, end=3)
        assert span.start == 0
        assert span.end == 3
        assert isinstance(span, tuple)


class TestEnums(unittest.TestCase):
    """Test Enum classes from tokenization_utils_base."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_padding_strategy_values(self):
        """test PaddingStrategy enum values."""
        assert PaddingStrategy.LONGEST.value == "longest"
        assert PaddingStrategy.MAX_LENGTH.value == "max_length"
        assert PaddingStrategy.DO_NOT_PAD.value == "do_not_pad"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_padding_strategy_from_string(self):
        """test PaddingStrategy from string."""
        strategy = PaddingStrategy("longest")
        assert strategy == PaddingStrategy.LONGEST

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_truncation_strategy_values(self):
        """test TruncationStrategy enum values."""
        assert TruncationStrategy.ONLY_FIRST.value == "only_first"
        assert TruncationStrategy.ONLY_SECOND.value == "only_second"
        assert TruncationStrategy.LONGEST_FIRST.value == "longest_first"
        assert TruncationStrategy.DO_NOT_TRUNCATE.value == "do_not_truncate"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_truncation_strategy_from_string(self):
        """test TruncationStrategy from string."""
        strategy = TruncationStrategy("only_first")
        assert strategy == TruncationStrategy.ONLY_FIRST

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_tensor_type_values(self):
        """test TensorType enum values."""
        assert TensorType.NUMPY.value == "np"
        assert TensorType.MINDSPORE.value == "ms"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_explicit_enum_error(self):
        """test ExplicitEnum raises error for invalid value."""
        with pytest.raises(ValueError) as exc_info:
            PaddingStrategy("invalid_value")
        assert "invalid_value" in str(exc_info.value)
        assert "not a valid" in str(exc_info.value)


class TestPaddingTruncationStrategies(unittest.TestCase):
    """Test padding and truncation strategy handling in base class."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())
        cls.path = cls.temp_dir
        get_sp_vocab_model("llama2_7b_test", cls.path)
        tokenizer_model_path = os.path.join(cls.path, "llama2_7b_test_tokenizer.model")
        cls.tokenizer = LlamaTokenizer(tokenizer_model_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_padding_truncation_strategies_defaults(self):
        """test _get_padding_truncation_strategies with default values."""
        padding_strategy, truncation_strategy, max_length, kwargs = \
            self.tokenizer._get_padding_truncation_strategies()
        assert padding_strategy == PaddingStrategy.DO_NOT_PAD
        assert truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE
        assert max_length is None
        assert kwargs == {}

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_padding_truncation_with_max_length_only(self):
        """test strategy inference when only max_length is provided."""
        _, truncation_strategy, max_length, _ = \
            self.tokenizer._get_padding_truncation_strategies(
                max_length=128, verbose=False
            )
        assert truncation_strategy == TruncationStrategy.LONGEST_FIRST
        assert max_length == 128

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_padding_max_length_strategy(self):
        """test padding max_length strategy."""
        padding_strategy, _, max_length, _ = \
            self.tokenizer._get_padding_truncation_strategies(
                padding='max_length', max_length=256
            )
        assert padding_strategy == PaddingStrategy.MAX_LENGTH
        assert max_length == 256

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_truncation_strategies(self):
        """test different truncation strategies."""
        # Test only_first
        _, truncation_strategy, _, _ = \
            self.tokenizer._get_padding_truncation_strategies(
                truncation='only_first', max_length=128
            )
        assert truncation_strategy == TruncationStrategy.ONLY_FIRST

        # Test only_second
        _, truncation_strategy, _, _ = \
            self.tokenizer._get_padding_truncation_strategies(
                truncation='only_second', max_length=128
            )
        assert truncation_strategy == TruncationStrategy.ONLY_SECOND

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pad_to_multiple_of(self):
        """test pad_to_multiple_of parameter."""
        padding_strategy, _, _, _ = \
            self.tokenizer._get_padding_truncation_strategies(
                padding=True, pad_to_multiple_of=8
            )
        assert padding_strategy == PaddingStrategy.LONGEST

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestTruncationEdgeCases(unittest.TestCase):
    """Test truncation edge cases in base class."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())
        cls.path = cls.temp_dir
        get_sp_vocab_model("llama2_7b_trunc", cls.path)
        tokenizer_model_path = os.path.join(cls.path, "llama2_7b_trunc_tokenizer.model")
        cls.tokenizer = LlamaTokenizer(tokenizer_model_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_truncate_left_side(self):
        """test left side truncation."""
        original_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ids_truncated, _, _ = self.tokenizer.truncate_sequences(
            ids, None, 5, TruncationStrategy.ONLY_FIRST, stride=0
        )
        assert len(ids_truncated) == 5
        assert ids_truncated == [6, 7, 8, 9, 10]
        self.tokenizer.truncation_side = original_side

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_truncate_right_side(self):
        """test right side truncation."""
        original_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "right"
        ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ids_truncated, _, _ = self.tokenizer.truncate_sequences(
            ids, None, 5, TruncationStrategy.ONLY_FIRST, stride=0
        )
        assert len(ids_truncated) == 5
        assert ids_truncated == [1, 2, 3, 4, 5]
        self.tokenizer.truncation_side = original_side

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_truncate_with_zero_removal(self):
        """test truncation when no tokens need to be removed."""
        ids = [1, 2, 3]
        ids_truncated, pair_truncated, overflow = self.tokenizer.truncate_sequences(
            ids, None, 0
        )
        assert ids_truncated == ids
        assert pair_truncated is None
        assert overflow == []

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_truncate_longest_first_with_pairs(self):
        """test longest_first truncation with pair sequences."""
        ids1 = [1, 2, 3, 4, 5]
        ids2 = [6, 7, 8]
        ids1_trunc, ids2_trunc, _ = self.tokenizer.truncate_sequences(
            ids1, ids2, 4, TruncationStrategy.LONGEST_FIRST
        )
        # Should remove from longest sequence first
        total_len = len(ids1_trunc) + len(ids2_trunc)
        assert total_len == 4

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestSpecialTokensMixin(unittest.TestCase):
    """Test SpecialTokensMixin class methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mixin = SpecialTokensMixin()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_special_tokens_attributes(self):
        """test SpecialTokensMixin has correct attributes."""
        assert hasattr(self.mixin, '_bos_token')
        assert hasattr(self.mixin, '_eos_token')
        assert hasattr(self.mixin, '_unk_token')
        assert hasattr(self.mixin, '_sep_token')
        assert hasattr(self.mixin, '_pad_token')
        assert hasattr(self.mixin, '_cls_token')
        assert hasattr(self.mixin, '_mask_token')
        assert hasattr(self.mixin, '_additional_special_tokens')

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_special_tokens_init_with_kwargs(self):
        """test SpecialTokensMixin initialization with kwargs."""
        mixin = SpecialTokensMixin(
            bos_token="<bos>",
            eos_token="<eos>",
            unk_token="<unk>"
        )
        assert mixin.bos_token == "<bos>"
        assert mixin.eos_token == "<eos>"
        assert mixin.unk_token == "<unk>"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_special_tokens_setter(self):
        """test SpecialTokensMixin token setters."""
        self.mixin.bos_token = "<s>"
        assert self.mixin._bos_token == "<s>"
        self.mixin.eos_token = "</s>"
        assert self.mixin._eos_token == "</s>"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_special_tokens_setter_with_added_token(self):
        """test SpecialTokensMixin setter with AddedToken."""
        token = AddedToken("<pad>", special=True)
        self.mixin.pad_token = token
        assert str(self.mixin.pad_token) == "<pad>"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_special_tokens_setter_invalid_type(self):
        """test SpecialTokensMixin setter with invalid type."""
        with pytest.raises(ValueError) as exc_info:
            self.mixin.bos_token = 123
        assert "non-string" in str(exc_info.value).lower()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_additional_special_tokens_setter(self):
        """test additional_special_tokens setter."""
        tokens = ["<token1>", "<token2>"]
        self.mixin.additional_special_tokens = tokens
        assert self.mixin._additional_special_tokens == tokens

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_special_tokens_map_extended(self):
        """test special_tokens_map_extended property."""
        self.mixin.bos_token = "<s>"
        self.mixin.eos_token = "</s>"
        special_map = self.mixin.special_tokens_map_extended
        assert isinstance(special_map, dict)
        assert special_map.get('bos_token') == "<s>"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_all_special_tokens_extended(self):
        """test all_special_tokens_extended property."""
        self.mixin.bos_token = "<s>"
        self.mixin.eos_token = "</s>"
        all_tokens = self.mixin.all_special_tokens_extended
        assert isinstance(all_tokens, list)
        # Check no duplicates
        token_strs = [str(t) for t in all_tokens]
        assert len(token_strs) == len(set(token_strs))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_special_tokens_cache(self):
        """test reset_special_tokens_cache method."""
        self.mixin._all_special_tokens = ["<s>"]
        self.mixin._all_special_ids = [1]
        self.mixin.reset_special_tokens_cache()
        assert self.mixin._all_special_tokens == []
        assert self.mixin._all_special_ids == []

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_special_tokens_init_invalid_additional_tokens(self):
        """test SpecialTokensMixin init with invalid additional_special_tokens."""
        with pytest.raises(ValueError) as exc_info:
            SpecialTokensMixin(additional_special_tokens="not_a_list")
        assert "not a list or tuple" in str(exc_info.value)


class TestConvertAddedTokens(unittest.TestCase):
    """Test convert_added_tokens class method."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_added_token_dict_to_object(self):
        """test converting dict with __type to AddedToken."""
        token_dict = {
            "__type": "AddedToken",
            "content": "<pad>",
            "special": True,
            "normalized": False
        }
        result = PreTrainedTokenizerBase.convert_added_tokens(token_dict)
        assert isinstance(result, AddedToken)
        assert result.content == "<pad>"
        assert result.special is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_added_token_object_to_dict(self):
        """test converting AddedToken to dict."""
        token = AddedToken("<pad>", special=True)
        result = PreTrainedTokenizerBase.convert_added_tokens(token, save=True)
        assert isinstance(result, dict)
        assert result.get("content") == "<pad>"
        assert result.get("special") is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_added_tokens_list(self):
        """test converting list of added tokens."""
        tokens = [
            {"__type": "AddedToken", "content": "<pad>", "special": True},
            {"__type": "AddedToken", "content": "<unk>", "special": True}
        ]
        result = PreTrainedTokenizerBase.convert_added_tokens(tokens)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(t, AddedToken) for t in result)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_added_tokens_nested_dict(self):
        """test converting nested dict with added tokens."""
        data = {
            "special_tokens": {
                "pad": {"__type": "AddedToken", "content": "<pad>", "special": True}
            }
        }
        result = PreTrainedTokenizerBase.convert_added_tokens(data)
        assert isinstance(result["special_tokens"]["pad"], AddedToken)


class TestPadMethod(unittest.TestCase):
    """Test _pad method from base class."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())
        cls.path = cls.temp_dir
        get_sp_vocab_model("llama2_7b_pad", cls.path)
        tokenizer_model_path = os.path.join(cls.path, "llama2_7b_pad_tokenizer.model")
        cls.tokenizer = LlamaTokenizer(tokenizer_model_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pad_do_not_pad(self):
        """test _pad with DO_NOT_PAD strategy."""
        encoded = {"input_ids": [1, 2, 3]}
        result = self.tokenizer._pad(
            encoded,
            padding_strategy=PaddingStrategy.DO_NOT_PAD
        )
        assert result["input_ids"] == [1, 2, 3]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pad_right_side(self):
        """test _pad with right side padding."""
        original_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"

        encoded = {"input_ids": [1, 2, 3]}
        result = self.tokenizer._pad(
            encoded,
            max_length=5,
            padding_strategy=PaddingStrategy.MAX_LENGTH,
            return_attention_mask=True
        )
        assert len(result["input_ids"]) == 5
        assert result["input_ids"][-1] == self.tokenizer.pad_token_id
        assert result["attention_mask"] == [1, 1, 1, 0, 0]

        self.tokenizer.padding_side = original_side

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pad_left_side(self):
        """test _pad with left side padding."""
        original_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        encoded = {"input_ids": [1, 2, 3]}
        result = self.tokenizer._pad(
            encoded,
            max_length=5,
            padding_strategy=PaddingStrategy.MAX_LENGTH,
            return_attention_mask=True
        )
        assert len(result["input_ids"]) == 5
        assert result["input_ids"][0] == self.tokenizer.pad_token_id
        assert result["attention_mask"] == [0, 0, 1, 1, 1]

        self.tokenizer.padding_side = original_side

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pad_with_token_type_ids(self):
        """test _pad with token_type_ids."""
        encoded = {
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0]
        }
        result = self.tokenizer._pad(
            encoded,
            max_length=5,
            padding_strategy=PaddingStrategy.MAX_LENGTH
        )
        assert len(result["token_type_ids"]) == 5
        assert result["token_type_ids"][-1] == self.tokenizer.pad_token_type_id

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pad_to_multiple_of(self):
        """test _pad with pad_to_multiple_of."""
        encoded = {"input_ids": [1, 2, 3]}
        result = self.tokenizer._pad(
            encoded,
            max_length=5,
            padding_strategy=PaddingStrategy.MAX_LENGTH,
            pad_to_multiple_of=8
        )
        # Should pad to 8 (next multiple of 8)
        assert len(result["input_ids"]) == 8

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestCleanUpTokenization(unittest.TestCase):
    """Test clean_up_tokenization static method."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_clean_up_spaces_before_punctuation(self):
        """test cleaning spaces before punctuation."""
        text = "Hello , world ."
        result = PreTrainedTokenizerBase.clean_up_tokenization(text)
        assert result == "Hello, world."

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_clean_up_contractions(self):
        """test cleaning contractions."""
        text = "I do n't know what 's happening"
        result = PreTrainedTokenizerBase.clean_up_tokenization(text)
        assert "n't" in result
        assert "'s" in result
        assert " n't" not in result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_clean_up_multiple_issues(self):
        """test cleaning multiple tokenization issues."""
        text = "It 's a test ! Do n't worry ."
        result = PreTrainedTokenizerBase.clean_up_tokenization(text)
        assert result == "It's a test! Don't worry."


class TestBatchEncodingConversion(unittest.TestCase):
    """Test BatchEncoding tensor conversion."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_already_tensor(self):
        """test that already-tensor data is not converted again."""
        data = {'input_ids': ms.Tensor([1, 2, 3])}
        encoding = BatchEncoding(data, tensor_type=TensorType.MINDSPORE)
        assert isinstance(encoding['input_ids'], ms.Tensor)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_none_tensor_type(self):
        """test with None tensor type."""
        data = {'input_ids': [1, 2, 3]}
        encoding = BatchEncoding(data, tensor_type=None)
        assert isinstance(encoding['input_ids'], list)


class TestFromExperimentalPretrained(unittest.TestCase):
    """Test from_experimental_pretrained method."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())
        cls.path = cls.temp_dir
        # Create a simple tokenizer for testing
        get_sp_vocab_model("llama2_7b_exp", cls.path)
        cls.tokenizer_model_path = os.path.join(cls.path, "llama2_7b_exp_tokenizer.model")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_experimental_pretrained_with_config_json(self):
        """test from_experimental_pretrained with tokenizer_config.json."""
        # Create tokenizer config
        config = {
            "tokenizer_class": "LlamaTokenizer",
            "model_max_length": 2048,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<unk>"
        }
        config_path = os.path.join(self.path, "tokenizer_config.json")
        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config, f)
        # Copy tokenizer model
        tokenizer_dest = os.path.join(self.path, "tokenizer.model")
        if not os.path.exists(tokenizer_dest):
            shutil.copy(self.tokenizer_model_path, tokenizer_dest)

        # Load tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(self.path)
        assert tokenizer is not None
        assert tokenizer.model_max_length == 2048

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_experimental_pretrained_with_special_tokens_map(self):
        """test from_experimental_pretrained with special_tokens_map.json."""
        # Create special tokens map
        special_tokens = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>"
        }
        special_tokens_path = os.path.join(self.path, "special_tokens_map.json")
        with open(special_tokens_path, "w", encoding="utf-8") as f:
            json.dump(special_tokens, f)

        config = {"tokenizer_class": "LlamaTokenizer"}
        config_path = os.path.join(self.path, "tokenizer_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        # Copy tokenizer model
        tokenizer_dest = os.path.join(self.path, "tokenizer.model")
        if not os.path.exists(tokenizer_dest):
            shutil.copy(self.tokenizer_model_path, tokenizer_dest)

        tokenizer = LlamaTokenizer.from_pretrained(self.path)
        assert tokenizer is not None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_experimental_pretrained_nonexistent_dir(self):
        """test from_experimental_pretrained with nonexistent directory."""
        with pytest.raises(Exception):
            LlamaTokenizer.from_pretrained("/nonexistent/path/to/tokenizer")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestSaveExperimentalPretrained(unittest.TestCase):
    """Test save_experimental_pretrained method."""

    @classmethod
    def setUpClass(cls):
        cls._exit_stack = ExitStack()
        cls.temp_dir = cls._exit_stack.enter_context(tempfile.TemporaryDirectory())
        cls.path = cls.temp_dir
        get_sp_vocab_model("llama2_7b_save", cls.path)
        tokenizer_model_path = os.path.join(cls.path, "llama2_7b_save_tokenizer.model")
        cls.tokenizer = LlamaTokenizer(tokenizer_model_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_experimental_pretrained_basic(self):
        """test save_experimental_pretrained saves required files."""
        save_dir = os.path.join(self.path, "saved_tokenizer")
        os.makedirs(save_dir, exist_ok=True)

        self.tokenizer.save_pretrained(save_dir, save_json=True)

        # Check that tokenizer_config.json exists
        config_path = os.path.join(save_dir, "tokenizer_config.json")
        assert os.path.exists(config_path)

        # Check that special_tokens_map.json exists
        special_tokens_path = os.path.join(save_dir, "special_tokens_map.json")
        assert os.path.exists(special_tokens_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_experimental_pretrained_with_prefix(self):
        """test save_experimental_pretrained with filename_prefix."""
        save_dir = os.path.join(self.path, "saved_with_prefix")
        os.makedirs(save_dir, exist_ok=True)

        self.tokenizer.save_pretrained(
            save_dir,
            save_json=True,
            filename_prefix="my_tokenizer"
        )

        # Check that files with prefix exist
        config_path = os.path.join(save_dir, "my_tokenizer-tokenizer_config.json")
        assert os.path.exists(config_path)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_and_load_roundtrip(self):
        """test save and load round trip."""
        save_dir = os.path.join(self.path, "roundtrip")
        os.makedirs(save_dir, exist_ok=True)

        # Save
        self.tokenizer.save_pretrained(save_dir, save_json=True)

        # Load
        loaded_tokenizer = LlamaTokenizer.from_pretrained(save_dir)

        # Compare
        test_text = "Hello world"
        original_encoding = self.tokenizer.encode(test_text)
        loaded_encoding = loaded_tokenizer.encode(test_text)
        assert original_encoding == loaded_encoding

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, '_exit_stack'):
            cls._exit_stack.close()


class TestIsExperimentalMode(unittest.TestCase):
    """Test is_experimental_mode utility function."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_is_experimental_mode_with_json(self):
        """test is_experimental_mode returns True with json config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = temp_dir
            # Create config.json
            config_path = os.path.join(path, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"model_type": "llama"}, f)

            result = is_experimental_mode(path)
            assert result is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_is_experimental_mode_with_yaml(self):
        """test is_experimental_mode returns False with yaml config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = temp_dir

            # Create yaml file
            yaml_path = os.path.join(path, "model.yaml")
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump({"model_type": "llama"}, f)

            result = is_experimental_mode(path)
            assert result is False

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_is_experimental_mode_with_tokenizer_config(self):
        """test is_experimental_mode with tokenizer_config.json."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = temp_dir
            # Create tokenizer_config.json
            config_path = os.path.join(path, "tokenizer_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"tokenizer_class": "LlamaTokenizer"}, f)

            result = is_experimental_mode(path)
            assert result is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_is_experimental_mode_nonexistent_path(self):
        """test is_experimental_mode with nonexistent path."""
        result = is_experimental_mode("/nonexistent/path")
        assert result is True  # Should return True for non-URL paths


def create_yaml(model_name, dir_path):
    """create yaml."""
    yaml_content = {
        "processor": {
            "return_tensors": "ms",
            "tokenizer": {
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<unk>",
                "type": "LlamaTokenizer"
            },
            "type": "LlamaProcessor"
        }
    }
    file_name = f'{dir_path}/{model_name}.yaml'
    with open(file_name, "w", encoding="utf-8") as file:
        yaml.dump(yaml_content, file, default_flow_style=False)
