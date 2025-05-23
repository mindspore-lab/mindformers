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
"""Test HF tokenizer."""
import os
import json
import unittest
import tempfile
import pytest
import tokenizers
from tokenizers import Tokenizer, pre_tokenizers, processors, decoders, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from transformers import AutoTokenizer

def _create_tokenizer_json(model_path):
    """Create tokenizer.json and tokenizer_config.json."""
    tokens = ["<unk>", "<s>", "</s>", "<pad>"]
    special_tokens = []
    for token in tokens:
        special_tokens.append(AddedToken(content=token, special=True))
    tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(pattern=tokenizers.Regex(" ?[^(\\s|[.,!?…。，、।۔،])]+"),
                              behavior="isolated", invert=False),
         pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    string = """
            华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。
            An increasing sequence: one, two, three, five, six, seven, nine, 10, 11, 12, 13.
            """

    trainer = BpeTrainer(vocab_size=200, special_tokens=special_tokens)
    tokenizer.train_from_iterator([string], trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.save(os.path.join(model_path, "tokenizer.json"))

    config = {
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "151643": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
                },
            "151644": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
                },
            "151645": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
                }
        },
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        "chat_template": "{% for message in messages %}{% if loop.first and "
                         "messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are "
                         "a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + "
                         "message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
                         "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|endoftext|>",
        "errors": "replace",
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "split_special_tokens": False,
        "tokenizer_class": "Qwen2Tokenizer",
    }

    with open(os.path.join(model_path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


class TestHfTokenizer(unittest.TestCase):
    """A test class for testing HF tokenizer"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        _create_tokenizer_json(cls.path)
        cls.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=cls.path)
        cls.string = "An increasing sequence: one, two, three."
        cls.input_ids = [114, 164, 117, 115, 192, 107, 124, 197, 13, 168, 4, 103, 127, 4, 103, 119, 17, 5]
        cls.attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_call_method(self):
        res = self.tokenizer(self.string)
        assert res.input_ids == self.input_ids
        assert res.attention_mask == self.attention_mask

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_tokenize(self):
        res = self.tokenizer.tokenize(self.string)
        assert res == ['An', 'Ġin', 'cre', 'as', 'ing', 'Ġse', 'qu', 'ence', ':',
                       'Ġone', ',', 'Ġt', 'wo', ',', 'Ġt', 'hre', 'e', '.']

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_pad(self, max_length=20):
        res = self.tokenizer(self.string, max_length=max_length, padding="max_length")
        assert res.input_ids == [114, 164, 117, 115, 192, 107, 124, 197, 13, 168, 4, 103,
                                 127, 4, 103, 119, 17, 5, 200, 200]
        assert res.attention_mask == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_encode(self):
        res = self.tokenizer.encode(self.string)
        assert res == [114, 164, 117, 115, 192, 107, 124, 197, 13, 168, 4, 103, 127, 4, 103, 119, 17, 5]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_decode(self):
        res = self.tokenizer.decode(self.input_ids)
        assert res == self.string

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_batch_decode(self):
        input_ids = [self.input_ids] * 10
        string = [self.string] * 10
        res = self.tokenizer.batch_decode(input_ids)
        assert res == string

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_vocab(self):
        res = self.tokenizer.get_vocab()
        assert len(res) == 206

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_convert_tokens_to_ids(self):
        res = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.string))
        assert res == [114, 164, 117, 115, 192, 107, 124, 197, 13, 168, 4, 103, 127, 4, 103, 119, 17, 5]

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_tokens(self):
        num_added_toks = self.tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        assert num_added_toks == 2

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_special_tokens(self):
        special_tokens_dict = {"cls_token": "<CLS>"}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        assert self.tokenizer.cls_token == "<CLS>"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_apply_chat_template(self):
        """Test HF tokenizer apply_chat_template interface."""
        # Define a conversation history
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        assert "Hello" in prompt
        assert "Hi there" in prompt
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        assert "Hello" in prompt
        assert "Hi there" in prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        assert "You are a helpful assistant." in prompt
