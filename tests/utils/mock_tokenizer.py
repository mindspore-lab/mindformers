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
"""
Mock Tokenizer.
"""

import re
import jieba


class MockTokenizer:
    """
    A simple mock tokenizer implementation for demonstration and testing purposes.

    This tokenizer provides a lightweight simulation of behavior similar to HuggingFace tokenizers.
    It supports:
    - Dynamic vocabulary building: new tokens are added when encountered.
    - Tokenization of plain text and text with special angle-bracket tokens (e.g., <start>, </end>).
    - Decoding from token IDs back to text.
    - Batch processing of multiple input strings.
    - Integration with jieba for Chinese word segmentation.
    """

    def __init__(self, **kwargs):
        """
        Initialize the MockTokenizer.
        Sets up empty vocabulary and inverse vocabulary dictionaries.
        """
        _ = kwargs
        self.vocab = dict()
        self.inv_vocab = dict()
        self.vocab_size = 0

    def __call__(self, text, **kwargs):
        """
        Tokenize the input text(s) and return input_ids and attention_mask.
        Supports both single string and list of strings as input.
        """
        _ = kwargs
        if isinstance(text, list):
            tokens, input_ids, attention_mask = [], [], []
            for sub_text in text:
                cur_tokens, cur_input_ids = self.encode(sub_text)
                tokens.append(cur_tokens)
                input_ids.append(cur_input_ids)
                attention_mask.append([1] * len(cur_input_ids))
        else:
            tokens, input_ids = self.encode(text)
            attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def decode(self, input_ids, **kwargs):
        """
        Convert a list of input_ids back to a string using the inverse vocabulary.
        Unknown ids are replaced with unk_token.
        """
        _ = kwargs
        return " ".join([self.inv_vocab.get(i, self.unk_token) for i in input_ids])

    def encode(self, text, **kwargs):
        """
        Tokenize the input text and convert tokens to input_ids.
        Adds new tokens to the vocabulary if not already present.
        """
        _ = kwargs
        tokens = self._split_text(text)
        input_ids = []

        for token in tokens:
            if token not in self.vocab:
                self._update_token(token)
            input_ids.append(self.vocab[token])

        return tokens, input_ids

    def add_tokens(self, *args, **kwargs):
        """
        Add a list of tokens to the vocabulary.
        """
        _ = kwargs
        tokens = args[0]
        for token in tokens:
            self._update_token(token)

    def _update_token(self, token):
        """
        Add a single token to the vocabulary and update the inverse vocabulary.
        """
        self.vocab_size += 1
        self.vocab[token] = self.vocab_size
        self.inv_vocab[self.vocab_size] = token

    @staticmethod
    def _split_text(text: str):
        """
        Split the input text into tokens.
        Text inside angle brackets is treated as a single token.
        Other text is segmented using jieba.
        """
        if not isinstance(text, str):
            raise ValueError("input text only support 'str' instance.")
        text = re.findall(r'<[^>]+>|[^<]+', text)
        words = list()
        for item in text:
            if item.startswith('<') and item.endswith('>'):
                words.append(item)
            else:
                words.extend([str(_) for _ in jieba.cut(item)])
        return words
