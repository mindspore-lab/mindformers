# coding: utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Telechat tokenizer APIs."""
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional

import sentencepiece as spm

from mindformers.tools import logger
from mindformers.models.tokenization_utils import PreTrainedTokenizer, AddedToken
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.utils import FILE_PERMISSION


VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class TelechatTokenizer(PreTrainedTokenizer):
    r"""
    Tokenize the input string and convert them into the ids. The tokenizer use the sentence piece internally.

    Args:
        model_path(str): The spiece.model file path.
        add_bos(bool): The flag defines whether add bos token, Default True.
        eos_token(str): The token that represents the end-of-sentence. Default "</s>".
        unk_token(str): The token that represents the unknown. Default "<unk>".
        pad_token(str): The token that represents the pad. Default "<pad>".
        sp_model_kwargs(str): Other kwargs for sp_model`.
        add_bos_token(bool): Whether or not to add the bos_token_id to the left of the input. Default "True"
        add_eos_token(bool): Whether or not to add the eos_token_id to the right of the input. Default "True"
        clean_up_tokenization_spaces (bool): Whether or not the model should cleanup the spaces that were added when
        splitting the input text during the tokenization process.  Default "False"
        **kwargs: Other kwargs that will be passed into the base class of the `Tokenizer`.

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
            bos_token="<_start>",
            eos_token="<_end>",
            pad_token="<_pad>",
            usr_token="<_user>",
            bot_token="<_bot>",
            sys_token="<_system>",
            call_start_token="<tool_call>",
            call_end_token="</tool_call>",
            repo_start_token="<tool_response>",
            repo_end_token="</tool_response>",
            chat_template=None,
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            add_bos_token=False,
            add_eos_token=False,
            clean_up_tokenization_spaces=False,
            **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False, single_word=False, normalized=False) \
            if isinstance(unk_token, str) else unk_token
        bos_token = AddedToken(bos_token, \
            lstrip=False, rstrip=False, single_word=False, normalized=False, special=True)
        eos_token = AddedToken(eos_token, \
            lstrip=False, rstrip=False, single_word=False, normalized=False, special=True)
        pad_token = AddedToken(pad_token, \
            lstrip=False, rstrip=False, single_word=False, normalized=False, special=True)
        usr_token = AddedToken(usr_token, \
            lstrip=False, rstrip=False, single_word=False, normalized=False, special=True)
        bot_token = AddedToken(bot_token, \
            lstrip=False, rstrip=False, single_word=False, normalized=False, special=True)
        sys_token = AddedToken(sys_token, \
            lstrip=False, rstrip=False, single_word=False, normalized=False, special=True)
        call_start_token = AddedToken(call_start_token, \
            lstrip=False, rstrip=False, single_word=False, normalized=False, special=True)
        call_end_token = AddedToken(call_end_token, \
            lstrip=False, rstrip=False, single_word=False, normalized=False, special=True)
        repo_start_token = AddedToken(repo_start_token, \
            lstrip=False, rstrip=False, single_word=False, normalized=False, special=True)
        repo_end_token = AddedToken(repo_end_token, \
            lstrip=False, rstrip=False, single_word=False, normalized=False, special=True)
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        self.chat_template = chat_template

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            chat_template=self.chat_template,
            **kwargs,
        )
        self.add_tokens([bos_token, eos_token, pad_token, usr_token, bot_token, sys_token, \
            call_start_token, call_end_token, repo_start_token, repo_end_token])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Returns a tokenized string."""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

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
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return None
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            with os.fdopen(os.open(out_vocab_file, flags_, FILE_PERMISSION), 'wb') as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return out_vocab_file

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
                                already_has_special_tokens: bool = False):
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

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
