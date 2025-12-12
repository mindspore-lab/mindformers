# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test tokenization utils."""
import os
import shutil
import sys
import pytest
from mindformers.models.tokenization_utils import (
    PreTrainedTokenizer,
    Trie,
    AddedToken,
    _is_whitespace,
    _is_control,
    _is_punctuation,
    _is_end_of_word,
    _is_start_of_word,
    _insert_one_token_to_ordered_list
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class ConcreteTokenizer(PreTrainedTokenizer):
    """ concrete tokenizer """
    def __init__(self, vocab_file, **kwargs):
        self.vocab = {}
        with open(vocab_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    token = line.strip()
                    self.vocab[token] = len(self.vocab)

        # Add UNK token if not present
        if "<unk>" not in self.vocab:
            self.vocab["<unk>"] = len(self.vocab)

        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        super().__init__(unk_token="<unk>", **kwargs)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text):
        # Simple whitespace tokenizer for testing
        return text.split()

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)


class TestTokenizationUtils:
    """ test tokenization utils """
    @classmethod
    def setup_class(cls):
        """ Create a temporary directory """
        cls.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_tokenization_utils_coverage")
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        os.makedirs(cls.test_dir)

        # Generate a simple vocab file
        cls.vocab_file = os.path.join(cls.test_dir, "vocab.txt")
        with open(cls.vocab_file, "w", encoding="utf-8") as f:
            vocab_list = ["hello", "world", "this", "is", "a", "test", "tokenizer", "[CLS]", "[SEP]", "[MASK]"]
            f.write("\n".join(vocab_list))

    @classmethod
    def teardown_class(cls):
        """ Remove the test directory """
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_trie_logic(self):
        """Test Trie add and split functionality"""
        trie = Trie()
        # Test basic add and split
        trie.add("[CLS]")
        trie.add("extra_id_1")

        text = "[CLS] This is a extra_id_1 test"
        split_text = trie.split(text)
        assert split_text == ["[CLS]", " This is a ", "extra_id_1", " test"]

        # Test duplicate add (idempotent)
        trie.add("[CLS]")
        split_text_2 = trie.split(text)
        assert split_text == split_text_2

        # Test empty string add
        trie.add("")

        # Test matching longest first
        trie.add("extra_id_100")
        text_long = "extra_id_100 should match full not extra_id_1"
        split_long = trie.split(text_long)
        assert split_long[0] == "extra_id_100"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_trie_atom_methods(self):
        """Test static atom methods of Trie class"""
        # Test split_atom_1: Checks if current state has reached a token end ("")
        # Construct a state where one path has ended
        states = {0: {"": 1}, 5: {}}
        offsets = []
        text = "sample text"
        res_offsets = Trie.split_atom_1(states, text, offsets)
        # Should append start(0) and len(text) to offsets and break
        assert res_offsets == [0, 11]

        # Test split_atom_2: Reset or cleanup states
        # Case 1: Reset = True
        states = {0: {}, 1: {}}
        res_states = Trie.split_atom_2(reset=True, states=states)
        assert res_states == {}

        # Case 2: Reset = False, remove specific keys
        states = {0: "val0", 1: "val1", 2: "val2"}
        to_remove = {1}
        res_states = Trie.split_atom_2(reset=False, to_remove=to_remove, states=states)
        assert res_states == {0: "val0", 2: "val2"}

        # We want to hit: `if "" in looktrie_pointer:` -> update start, end, skip
        looktrie_pointer = {"": 1}
        states = {0: looktrie_pointer}

        res_states, res_start, res_end, res_skip = Trie.split_atom_3(states, current=0, text="ab", start=0, skip=0)
        assert res_start == 0
        # end is updated to lookahead_index which starts at current(0)
        assert res_end == 0
        assert res_skip == 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_module_helper_functions(self):
        """Test standalone helper functions in tokenization_utils"""
        # _is_whitespace
        assert _is_whitespace(" ")
        assert _is_whitespace("\t")
        assert _is_whitespace("\n")
        assert _is_whitespace("\r")
        assert _is_whitespace("\u00A0")
        assert not _is_whitespace("a")

        # _is_control
        # \t, \n, \r are NOT control in this specific function logic (explicitly excluded)
        assert not _is_control("\t")
        assert not _is_control("\n")
        assert not _is_control("\r")
        assert _is_control("\x00")  # Null char is control (Cc)
        assert not _is_control("a")

        # _is_punctuation
        assert _is_punctuation("!")
        assert _is_punctuation(",")
        # ASCII non-letter/number characters check: 33-47, 58-64, 91-96, 123-126
        assert _is_punctuation("$")  # 36
        assert not _is_punctuation("A")

        # _is_end_of_word: checks last char
        assert _is_end_of_word("word.")
        assert not _is_end_of_word("word")

        # _is_start_of_word: checks first char
        assert _is_start_of_word(".word")
        assert not _is_start_of_word("word")

        # _insert_one_token_to_ordered_list
        token_list = ["a", "c"]
        _insert_one_token_to_ordered_list(token_list, "b")
        assert token_list == ["a", "b", "c"]

        # Insert existing
        _insert_one_token_to_ordered_list(token_list, "b")
        assert token_list == ["a", "b", "c"]  # No duplicate

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_tokenizer_instantiation(self):
        """Test tokenizer initialization and vocab loading"""
        tokenizer = ConcreteTokenizer(self.vocab_file)
        assert tokenizer.vocab_size > 0
        assert "hello" in tokenizer.get_vocab()
        assert tokenizer.unk_token == "<unk>"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_tokenize_basic(self):
        """Test basic tokenization flow"""
        tokenizer = ConcreteTokenizer(self.vocab_file)
        text = "hello world"
        tokens = tokenizer.tokenize(text)
        assert tokens == ["hello", "world"]

        # Test with unknown token
        text_unk = "hello unknown_word"
        tokens_unk = tokenizer.tokenize(text_unk)
        assert tokens_unk == ["hello", "unknown_word"]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_convert_tokens_to_ids(self):
        """Test converting tokens to IDs"""
        tokenizer = ConcreteTokenizer(self.vocab_file)
        ids = tokenizer.convert_tokens_to_ids(["hello", "world"])
        assert isinstance(ids, list)
        assert len(ids) == 2
        assert ids[0] == tokenizer.vocab["hello"]

        # Single string input
        id_single = tokenizer.convert_tokens_to_ids("hello")
        assert id_single == tokenizer.vocab["hello"]

        # None input
        assert tokenizer.convert_tokens_to_ids(None) is None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_convert_ids_to_tokens(self):
        """Test converting IDs to tokens"""
        tokenizer = ConcreteTokenizer(self.vocab_file)
        target_id = tokenizer.vocab["hello"]
        tokens = tokenizer.convert_ids_to_tokens([target_id])
        assert tokens == ["hello"]

        # Single int input
        token_single = tokenizer.convert_ids_to_tokens(target_id)
        assert token_single == "hello"

        # Out of vocab size
        with pytest.raises(IndexError):
            tokenizer.convert_ids_to_tokens(99999)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_special_tokens_handling(self):
        """Test handling of added special tokens"""
        tokenizer = ConcreteTokenizer(self.vocab_file)

        # Add a new special token
        new_token = AddedToken("[SPECIAL]", special=True)
        tokenizer.add_special_tokens({"additional_special_tokens": [new_token]})

        assert "[SPECIAL]" in tokenizer.get_added_vocab()

        # Test encoding with special token not splitting
        text = "hello [SPECIAL] world"
        tokens = tokenizer.tokenize(text)
        assert "[SPECIAL]" in tokens

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_added_tokens_decoder_setter_validation(self):
        """Test validation logic in added_tokens_decoder setter"""
        tokenizer = ConcreteTokenizer(self.vocab_file)

        # Valid setter
        valid_dict = {100: AddedToken("token")}
        tokenizer.added_tokens_decoder = valid_dict
        assert tokenizer.added_tokens_decoder[100].content == "token"

        valid_dict_str = {101: "token_str"}
        tokenizer.added_tokens_decoder = valid_dict_str
        assert tokenizer.added_tokens_decoder[101].content == "token_str"

        # Invalid Key Type
        with pytest.raises(ValueError):
            tokenizer.added_tokens_decoder = {"bad_key": "val"}

        # Invalid Value Type
        with pytest.raises(ValueError):
            tokenizer.added_tokens_decoder = {102: 12345}

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_encode_decode_cycle(self):
        """Test full encode and decode cycle"""
        tokenizer = ConcreteTokenizer(self.vocab_file)
        text = "hello world"

        # Encode
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(input_ids) == 2

        # Decode
        decoded_text = tokenizer.decode(input_ids)
        assert decoded_text.strip() == text  # strip because join might add spaces

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_batch_encode_plus(self):
        """Test batch encoding"""
        tokenizer = ConcreteTokenizer(self.vocab_file)
        batch_text = ["hello world", "this is a test"]

        encoded = tokenizer.batch_encode_plus(batch_text, padding=False)
        assert len(encoded["input_ids"]) == 2
        assert len(encoded["input_ids"][0]) == 2  # hello world
        assert len(encoded["input_ids"][1]) == 4  # this is a test

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_num_special_tokens_to_add(self):
        """Test calculation of added special tokens"""
        tokenizer = ConcreteTokenizer(self.vocab_file)
        # Default build_inputs_with_special_tokens adds nothing if not overridden,
        # unless we set bos/eos/etc.
        count = tokenizer.num_special_tokens_to_add(pair=False)
        assert count == 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_prepare_for_tokenization(self):
        """Test hook for preparing text"""
        tokenizer = ConcreteTokenizer(self.vocab_file)
        text = "Raw Text"
        processed_text, kwargs = tokenizer.prepare_for_tokenization(text, custom_arg="val")
        assert text == processed_text
        assert kwargs == {"custom_arg": "val"}

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_advanced_input_handling(self):
        """Test advanced input types for coverage of _encode_plus"""
        tokenizer = ConcreteTokenizer(self.vocab_file)

        # 1. Test list of strings with is_split_into_words=True
        # Input: ["hello", "world"] -> treated as words, tokenized individually -> ["hello", "world"] -> ids
        res = tokenizer.encode_plus(["hello", "world"], is_split_into_words=True, return_attention_mask=False,
                                    return_token_type_ids=False)
        assert res['input_ids'] == [tokenizer.vocab["hello"], tokenizer.vocab["world"]]

        # 2. Test list of strings with is_split_into_words=False
        # Input: ["hello", "world"] -> treated as already tokenized strings
        res = tokenizer.encode_plus(["hello", "world"], is_split_into_words=False, return_attention_mask=False,
                                    return_token_type_ids=False)
        assert res['input_ids'] == [tokenizer.vocab["hello"], tokenizer.vocab["world"]]

        # 3. Test list of integers (pre-tokenized IDs)
        ids = [tokenizer.vocab["hello"], tokenizer.vocab["world"]]
        res = tokenizer.encode_plus(ids, return_attention_mask=False, return_token_type_ids=False)
        assert res['input_ids'] == ids

        # 4. Error: is_split_into_words=True but input is invalid (e.g. integer)
        with pytest.raises(ValueError):
            tokenizer.encode_plus(123, is_split_into_words=True)

        # 5. Error: Input not valid (e.g. float) and is_split_into_words=False
        with pytest.raises(ValueError):
            tokenizer.encode_plus(12.34)

        # 6. Error: return_offsets_mapping=True
        with pytest.raises(NotImplementedError):
            tokenizer.encode_plus("hello", return_offsets_mapping=True)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_batch_encode_plus_advanced(self):
        """Test advanced inputs for batch_encode_plus"""
        tokenizer = ConcreteTokenizer(self.vocab_file)

        # 1. List of list of strings (batch of pre-tokenized sentences)
        batch_tokens = [["hello", "world"], ["test", "tokenizer"]]
        # is_split_into_words=True means each list item is a sequence of words
        res = tokenizer.batch_encode_plus(batch_tokens, is_split_into_words=True, return_attention_mask=False)
        assert len(res['input_ids']) == 2
        assert res['input_ids'][0] == [tokenizer.vocab["hello"], tokenizer.vocab["world"]]

        # 2. List of list of ints (batch of IDs)
        # Note: is_split_into_words=True is required here because otherwise [id1, id2] is interpreted
        # as a pair (id1, id2), and get_input_ids(id1) fails because single int input is not supported.
        ids = [tokenizer.vocab["hello"], tokenizer.vocab["world"]]
        batch_ids = [ids, ids]
        res = tokenizer.batch_encode_plus(batch_ids, is_split_into_words=True, return_attention_mask=False)
        assert res['input_ids'] == batch_ids

        # 3. Error: return_offsets_mapping=True
        with pytest.raises(NotImplementedError):
            tokenizer.batch_encode_plus(["hello"], return_offsets_mapping=True)

        # 4. Batch with pairs (list of tuples)
        pairs = [("hello", "world"), ("test", "tokenizer")]
        res = tokenizer.batch_encode_plus(pairs, return_attention_mask=False)
        assert len(res['input_ids']) == 2

        # 5. Test invalid batch input to trigger ValueError
        # Input list of floats
        with pytest.raises(ValueError):
            tokenizer.batch_encode_plus([12.34])
