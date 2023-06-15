# Copyright 2023 Huawei Technologies Co., Ltd
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
"""PanguAlpha Tokenzier"""

import os

import jieba
import sentencepiece as spm

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.base_tokenizer import Tokenizer
from ...mindformer_book import MindFormerBook


__all__ = ['PanguAlphaTokenizer']


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class PanguAlphaTokenizer(Tokenizer):
    r"""
    Tokenize the input string and convert them into the ids. The tokenizer use the sentence piece internally.

    Args:
        model_file(str): The vocabulary file path.
        unk_token(str): The token that represents the unknown. Default "<unk>".
        bos_token(str): The token that represents the begin-of-sentence. Default "<s>".
        eos_token(str): The token that represents the end-of-sentence. Default "<eod>".
        pad_token(str): The token that represents the pad. Default "<pad>".
        **kwargs: Other kwargs that will be passed into the base class of the `Tokenizer`.

    Examples:
        >>> from mindformers import PanguAlphaTokenizer

        >>> tokenizer = PanguAlphaTokenizer.from_pretrained("pangualpha_2_6b")
        >>> res = tokenizer("你好，今天天气不错。")
        >>> print(res)
        {'input_ids': [5772, 10, 465, 235, 464, 1123, 12], \
        'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], \
        'attention_mask': [1, 1, 1, 1, 1, 1, 1]}

    Outputs:
        A dict contains the processed ids, attention_mask that specific by the member `MODEL_INPUT_NAME`
        of the subclass.
    """
    VOCAB_FILES = {'merge_file': 'merges.txt', 'vocab_file': 'vocab.model'}
    FILE_LIST = ['tokenizer_config.json']
    MODEL_INPUT_NAME = ["input_ids", "token_type_ids", "attention_mask"]
    _support_list = MindFormerBook.get_tokenizer_support_list()['pangualpha']

    def __init__(self,
                 vocab_file,
                 unk_token="<unk>",
                 bos_token="<s>",
                 eos_token="<eod>",
                 pad_token="<pad>",
                 **kwargs):
        super(PanguAlphaTokenizer, self).__init__(
            unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, **kwargs
        )
        self.encoder = {}
        self.sp = spm.SentencePieceProcessor(model_file=vocab_file)

        for i in range(self.sp.get_piece_size()):
            self.encoder[self.sp.id_to_piece(i)] = i
        self.translator = str.maketrans(" \n", "\u2582\u2583")

    def tokenize(self, text):
        if not isinstance(text, str):
            raise ValueError("Text should be type str, but found type", type(text))
        return self._tokenize(text)

    def _tokenize(self, text, **kwargs):
        """ Tokenize a string using bpe encode. """
        seg_list = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        return seg_list

    def _convert_tokens_to_ids(self, tokens):
        """ the index of the tokens in the vocabulary. """
        new_seg = " ".join(tokens)
        return self.sp.encode(new_seg)

    def _convert_ids_to_tokens(self, ids):
        """ return the origin bpe tokens according to ids """
        return self.sp.id_to_piece(ids)

    def _convert_tokens_to_string(self, tokens):
        """ return a string according to the list of tokens"""
        return self.pangu_decode(tokens)

    def convert_tokens_to_string(self, tokens):
        """Convert the tokens to the string"""
        return self._convert_tokens_to_string(tokens)

    def process_tokens(self, text):
        r"""replace special tokens with space and \n"""
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text

    def pangu_encode(self, text):
        """pangu encode"""
        res = self.tokenize(text)
        return res

    def pangu_decode(self, tokens):
        """pangu decode"""
        text = self.sp.decode(tokens)
        return self.process_tokens(text)

    def save_vocabulary(self, save_directory, filename_prefix):
        """write the word to the files"""
        output_file_path = os.path.join(save_directory, filename_prefix)
        with open(output_file_path, 'w', encoding="utf8") as fp:
            for k in self.encoder:
                fp.write(k + '\n')
        return output_file_path

    @property
    def vocab_size(self):
        return len(self.encoder)
