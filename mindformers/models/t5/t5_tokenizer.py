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
"""T5 Tokenzier"""

import os
import shutil

import jieba
import sentencepiece as spm

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.base_tokenizer import Tokenizer
from mindformers.models.bert import BertTokenizer
from ...mindformer_book import MindFormerBook

__all__ = ["T5Tokenizer", "T5PegasusTokenizer"]


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class T5Tokenizer(Tokenizer):
    r"""
    Tokenize the input string and convert them into the ids. The tokenizer use the sentence piece internally.

    Args:
        vocab_file(str): The spiece.model file path.
        eos_token(str): The token that represents the end-of-sentence. Default "</s>".
        unk_token(str: The token that represents the unknown. Default "<unk>".
        pad_token(str): The token that represents the pad. Default "<pad>".
        **kwargs: Other kwargs that will be passed into the base class of the `Tokenizer`.

    Examples:
        >>> from mindformers import T5Tokenizer
        >>> tokenizer = T5Tokenizer.from_pretrained("t5_small")
        >>> res = tokenizer("hello world")
        >>> print(res)
        {'input_ids': [21820, 296, 1], 'attention_mask': [1, 1, 1]}
        >>> res = tokenizer("hello world", padding='max_length', max_length=10)
        >>> print(res)
        {'input_ids': [21820, 296, 1, 0, 0, 0, 0, 0, 0, 0],
         'attention_mask': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]}
        >>> res = tokenizer("hello world", add_special_tokens=False)
        >>> print(res)
        {'input_ids': [21820, 296], 'attention_mask': [1, 1]}
        >>> res = tokenizer("hello world", return_tensors='ms')
        >>> print(res)
        {'input_ids': Tensor(shape=[3], dtype=Int32, value= [21820,   296,     1]),
        'attention_mask': Tensor(shape=[3], dtype=Int32, value= [1, 1, 1])}
        >>> res = tokenizer(["hello world", "today is a good day"],
        ...                 max_length=7, padding='max_length', return_tensors='ms')
        >>> print(res)
        {'input_ids': Tensor(shape=[3], dtype=Int32, value= [21820,   296,     1]),
        'attention_mask': Tensor(shape=[3], dtype=Int32, value= [1, 1, 1])}

    Outputs:
        A dict contains the processed ids, attention_mask that specific by the member `MODEL_INPUT_NAME`
        of the subclass.
    """
    VOCAB_FILES = {'vocab_file': 'spiece.model'}
    FILE_LIST = ['tokenizer_config.json']
    MODEL_INPUT_NAME = ['input_ids', 'attention_mask']

    _extra_pattern = r'<extra_id_(\d+)>'
    _support_list = MindFormerBook.get_tokenizer_support_list()['t5']

    def __init__(self,
                 vocab_file: str,
                 eos_token: str = "</s>",
                 unk_token: str = "<unk>",
                 pad_token: str = "<pad>",
                 extra_ids: int = 100,
                 **kwargs):
        """
        Initialize the sentence piece model according to the model path
        """
        super(T5Tokenizer, self).__init__(eos_token=eos_token,
                                          unk_token=unk_token,
                                          pad_token=pad_token,
                                          extra_ids=extra_ids,
                                          **kwargs)
        self.extra_ids = extra_ids
        self.s = spm.SentencePieceProcessor(model_file=vocab_file)
        self.vocab_file = vocab_file

    def _tokenize(self, text, **kwargs):
        token_list = self.s.encode(text, out_type=str)
        return token_list

    def _convert_tokens_to_ids(self, input_tokens):
        if not input_tokens:
            raise ValueError(f"Input token {input_tokens} is None.")
        if isinstance(input_tokens, str):
            return self.s.piece_to_id(input_tokens)
        res = []
        for item in input_tokens:
            res.append(self.s.piece_to_id(item))
        return res

    def tokenize(self, text):
        """Tokenizer the input_text"""
        if not isinstance(text, str):
            raise ValueError("Text should be type str, but found type", type(text))
        return self._tokenize(text)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Add the eos to the token_ids0"""
        if not token_ids_1:
            return token_ids_0 + [self.eos_token_id]
        raise ValueError("Only token_ids_1=None is supported now.")

    def save_vocabulary(self, save_directory, filename_prefix):
        """write the word to the files"""
        output_file_path = os.path.join(save_directory, filename_prefix)
        shutil.copy(self.vocab_file, output_file_path)
        return output_file_path

    def _convert_ids_to_tokens(self, ids):
        """convert the given ids to the tokens"""
        def convert_ids(ids):
            if ids < self.s.vocab_size():
                return self.s.IdToPiece(ids)
            return self._extra_pattern.replace(r'(\d+)', str(ids - self.s.vocab_size() - 1))

        if isinstance(ids, int):
            return convert_ids(ids)

        if isinstance(ids, list):
            res = []
            for item in ids:
                res.append(convert_ids(item))
            return res
        raise TypeError(f"The type of ids should be int or list, but found {type(ids)}.")

    def convert_tokens_to_string(self, tokens):
        if not tokens:
            return ""
        return self.s.decode_pieces(tokens).strip()

    @property
    def vocab_size(self):
        """Return the vocab size of the tokenizer"""
        return self.s.vocab_size() + self.extra_ids


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class T5PegasusTokenizer(BertTokenizer):
    """Tokenizer for T5 Pegasus, which is suitable for Chinese dataset.
    Based on word granularity, if the word does not appear in the vocabulary file,
        then call BERT native Tokenizer.

    Args:
        vocab_file(str): Path to vocab file.
        pre_tokenizer(function): Default using jieba to cut words
    """

    def __init__(self,
                 vocab_file,
                 *args,
                 pre_tokenizer=lambda x: jieba.cut(x, HMM=False),
                 **kwargs):
        super(T5PegasusTokenizer, self).__init__(vocab_file, *args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, **kwargs):
        split_tokens = []
        for word in self.pre_tokenizer(text):
            if word in self.vocab_dict:
                split_tokens.append(word)
            else:
                split_tokens.extend(super()._tokenize(word))
        return split_tokens

    def convert_tokens_to_string(self, tokens: list):
        """Convert the tokens to the string"""
        return " ".join(tokens)

    def _decode(self, token_ids, skip_special_tokens=False, **kwargs):
        ids = self.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        return self.convert_tokens_to_string(ids)
