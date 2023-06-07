
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
"""LLaMA tokenizer APIs."""

import os
import shutil

import sentencepiece as spm

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.base_tokenizer import Tokenizer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class LlamaTokenizer(Tokenizer):
    r"""
    Tokenize the input string and convert them into the ids. The tokenizer use the sentence piece internally.
    Tokenizer of llama will default add bos at the beginning of tokens and add eos token on the tail of tokens.

    Args:
        model_path(str): The spiece.model file path.
        add_bos(bool): The flag defines whether add bos token, Default True.
        eos_token(str): The token that represents the end-of-sentence. Default "</s>".
        unk_token(str: The token that represents the unknown. Default "<unk>".
        pad_token(str): The token that represents the pad. Default "<pad>".
        **kwargs: Other kwargs that will be passed into the base class of the `Tokenizer`.

    Examples:
        >>> from mindformers import LlamaTokenizer
        >>> tokenizer = LlamaTokenizer.from_pretrained(name_or_path="/path/tokenizer_dir")
        >>> res = tokenizer("hello world")
        >>> print(res)
        {'input_ids': [1, 22172, 3186, 2]}
        >>> res = tokenizer("hello world", padding='max_length', max_length=10)
        >>> print(res)
        {'input_ids': [1, 22172, 3186, 2, 0, 0, 0, 0, 0, 0]}
        >>> res = tokenizer("hello world", return_tensors='ms')
        >>> print(res)
        {'input_ids': Tensor(shape=[3], dtype=Int32, value= [1, 22172, 3186, 2])}

    Outputs:
        A dict contains the processed ids, attention_mask that specific by the member `MODEL_INPUT_NAME`
        of the subclass.
    """
    VOCAB_FILES = {'vocab_file': 'tokenizer.model'}
    FILE_LIST = ['tokenizer_config.json']
    MODEL_INPUT_NAME = ['input_ids', 'attention_mask']
    _support_list = MindFormerBook.get_tokenizer_support_list()['llama']

    _extra_pattern = r'<extra_id_(\d+)>'

    def __init__(self,
                 vocab_file,
                 extra_ids=100,
                 add_bos=True,
                 eos_token='</s>',
                 bos_token='<s>',
                 unk_token='<unk>',
                 pad_token='<pad>',
                 **kwargs):
        """
        Initialize the sentence piece model according to the vocab file
        """
        self.s = spm.SentencePieceProcessor(model_file=vocab_file, add_bos=add_bos)
        self.n_words = self.s.vocab_size()

        super().__init__(eos_token=eos_token,
                         bos_token=bos_token,
                         unk_token=unk_token,
                         pad_token=pad_token,
                         extra_ids=extra_ids,
                         **kwargs)
        self.extra_ids = extra_ids
        self.vocab_file = vocab_file

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
            raise ValueError(
                "Text should be type str, but found type", type(text))
        token_list = self.s.encode(text, out_type=str)
        return token_list

    def _tokenize(self, text, **kwargs):
        return None

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
        raise TypeError(
            f"The type of ids should be int or list, but found {type(ids)}.")

    def convert_tokens_to_string(self, tokens):
        if not tokens:
            return ""
        return self.s.decode_pieces(tokens).strip()

    @property
    def vocab_size(self):
        """Return the vocab size of the tokenizer"""
        return self.s.vocab_size() + self.extra_ids
