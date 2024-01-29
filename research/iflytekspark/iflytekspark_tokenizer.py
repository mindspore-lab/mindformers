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
"""iFlytekSpark model tokenizer APIs."""
import os
import re
import glob
from typing import Any, Dict, List, Optional
from shutil import copyfile
import sentencepiece as spm
from mindformers.models.base_tokenizer import Tokenizer, AddedToken
from mindformers.tools import logger
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


class BaseTokenizer():
    """Base tokenizer class."""
    @property
    def eod(self):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, tokens):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SPTokenizer(BaseTokenizer):
    """Tokenizer process special tokens."""
    def __init__(self, file_path: str, sp_model_kwargs: Optional[Dict[str, Any]] = None):
        file_extension = [".model", ".vocab"]
        model_file = glob.glob(os.path.join(file_path, f"*{file_extension[0]}"))[0]
        vocab_file = glob.glob(os.path.join(file_path, f"*{file_extension[1]}"))[0]

        assert os.path.exists(vocab_file), \
            f"vocab file path ({vocab_file}) is not exist"

        assert os.path.exists(model_file), \
            f"vocab file path ({model_file}) is not exist"
        f = open(vocab_file, "r", encoding="utf-8")
        lines = f.readlines()
        self.encoder = {}
        for line in enumerate(lines):
            key = line[1].split("\t")[0]
            self.encoder[key] = line[0]
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.sp = spm.SentencePieceProcessor(**sp_model_kwargs)
        self.sp.Load(model_file)

        self.eod_id = self.encoder["<end>"]
        self.pad_id = self.encoder["<pad>"]
        self.unk_id = self.encoder["<unk>"]

    @property
    def vocab_size(self):
        return len(self.encoder)

    @property
    def eod(self):
        return self.eod_id

    def add_space(self, text):
        text = re.sub("(，|。|！|？) *", r"\1 ", text)
        return text

    def __len__(self):
        return len(self.encoder)

    def tokenize(self, text):
        text = text.replace("\n", "<ret>")
        text = text.replace("\t", " " * 4)
        return self.sp.encode(text)

    def convert_tokens_to_ids(self, tokens):
        return tokens

    def convert_ids_to_tokens(self, ids):
        return self.decode(ids)

    def decode_tokens(self, tokens, keep_special_token=True):
        text = self.sp.DecodePieces(tokens)
        if not keep_special_token:
            text = text.replace("<ret>", "\n")
        return text

    def detokenize(self, token_ids):
        return self.decode(token_ids)

    def encode(self, text):
        return self.tokenize(text)

    def decode(self, tokens):
        text = self.sp.decode(tokens)
        return text


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class IFlytekSparkTokenizer(Tokenizer):
    r"""
    Tokenize the input string and convert them into the ids. The tokenizer use the sentence piece internally.

    Args:
        vocab_file(str): The spiece.vocab file path.
        eos_token(str): The token that represents the end-of-sentence. Default "<end>".
        unk_token(str): The token that represents the unknown. Default "<unk>".
        pad_token(str): The token that represents the pad. Default "<pad>".
        sp_model_kwargs(str): Other kwargs for sp_model`.
        add_bos_token(bool): Whether or not to add the bos_token_id to the left of the input. Default "False"
        add_eos_token(bool): Whether or not to add the eos_token_id to the right of the input. Default "False"
        clean_up_tokenization_spaces (bool): Whether or not the model should cleanup the spaces that were added when
        splitting the input text during the tokenization process.  Default "False"
        **kwargs: Other kwargs that will be passed into the base class of the `Tokenizer`.

    Examples:
        >>> from mindformers.research.iflytekspark\iflytekspark_tokenizer import IFlytekSparkTokenizer
        >>> file_path = "{your_tokenizer_file_path}" # change to your local path
        >>> tokenizer = IFlytekSparkTokenizer("file_path")
        >>> res = tokenizer("今天天气很好")
        >>> print(res)
        {'input_ids': [1316, 4499, 3120], 'attention_mask': [1, 1, 1]}
        >>> res = tokenizer("hello world", return_tensors='ms')
        >>> print(res)
        {'input_ids': Tensor(shape=[3], dtype=Int32, value=[1316, 4499, 3120]), 'attention_mask': Tensor(shape=[3],
        dtype=Int32, value=[1, 1, 1])}
        >>> res = tokenizer.decode([1316, 4499, 3120])
        >>> print(res)
        '今天天气很好
'
    Outputs:
        A dict contains the processed ids, attention_mask that specific by the member `MODEL_INPUT_NAME`
        of the subclass.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    FILE_LIST = ['tokenizer_config.json']

    def __init__(
            self,
            vocab_file,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="<end>",
            pad_token="<pad>",
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            add_bos_token=False,
            add_eos_token=False,
            clean_up_tokenization_spaces=False,
            **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False, single_word=False, normalized=True) \
            if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False, single_word=True, normalized=True) \
            if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False, single_word=True, normalized=True) \
            if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False, single_word=True, normalized=True) \
            if isinstance(pad_token, str) else pad_token

        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model = SPTokenizer(self.vocab_file, self.sp_model_kwargs)

        super(IFlytekSparkTokenizer, self).__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = SPTokenizer(self.vocab_file, self.sp_model_kwargs)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.vocab_size

    @property
    def eod(self):
        return self.sp_model.eod

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # pylint: disable=W0613
    def tokenize(self, text, pair=None, add_special_tokens=True, **kwargs):
        """ Returns a tokenized string. """
        return self._tokenize(text)

    # pylint: disable=W0221
    # pylint: disable=W0613
    def encode(self, text, pair=None, add_special_tokens=True, **kwargs):
        """ Returns a tokenized string. """
        return self._tokenize(text)

    def _tokenize(self, text, **kwargs):
        """Returns a tokenized string."""
        return self.sp_model.encode(text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.convert_tokens_to_ids(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index >= self.vocab_size:
            raise IndexError(
                f"The token id {index} is out of the size of vocabulary, please check your tokenizer "
                f"and corresponding vocabulary files.")
        token = self.sp_model.convert_ids_to_tokens(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        return self.sp_model.decode_tokens(tokens)

    # pylint: disable=R1710
    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path %s should be a directory", save_directory)
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.sp.serialized_model_proto()
                fi.write(content_spiece_model)

        return out_vocab_file

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output


    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output
