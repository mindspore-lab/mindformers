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
"""Bese Tokenzier for the pretrained tokenizer"""
import os
import json

__all__ = ['PretrainedTokenizerBase', 'PretrainedTokenizer', 'SpecialTokensMixin']


SPECIAL_TOKEN_FILE_NAME = 'special_tokens_map.json'
VOCAB_FILE_NAME = 'vocab.txt'
TOKENIZER_CONFIG_NAME = 'tokenizer_config.json'

class SpecialTokensMixin:
    """A class for managing the specific tokens"""
    SPECIAL_TOKENS = ['pad_token', 'cls_token', 'sep_token', 'pad_token', 'mask_token']
    def __init__(self,
                 **kwargs):

        self._pad_token = None
        self._sep_token = None
        self._cls_token = None
        self._mask_token = None
        for k, v in kwargs.items():
            for item in self.SPECIAL_TOKENS:
                if item == k:
                    setattr(self, '_' + item, v)
    @property
    def pad_token(self):
        return self._pad_token

    @property
    def pad_token_id(self):
        return self._convert_tokens_to_ids(self._pad_token)

    @property
    def sep_token(self):
        return self._sep_token

    @property
    def sep_token_id(self):
        return self._convert_tokens_to_ids(self._sep_token)

    @property
    def cls_token(self):
        return self._cls_token

    @property
    def cls_token_id(self):
        return self._convert_tokens_to_ids(self._cls_token)


class PretrainedTokenizerBase(SpecialTokensMixin):
    """The pretrained tokenize providing basic method for tokenizing."""
    MODEL_INPUT_NAME = ['input_ids', 'attention_mask', 'token_type_ids']
    VOCAB_FILES = {}
    FILE_LIST = []
    def __init__(self, **kwargs):
        super(PretrainedTokenizerBase, self).__init__(**kwargs)
        self.model_inputs = self.MODEL_INPUT_NAME
        self.init_kwargs = kwargs

    def __call__(self,
                 text,
                 text_pair=None):
        """
        Get the text inputs and then renturn the padded ids

        It need three steps:
            1. tokenize
            2. convert them to ids
            3. combine them to a batch
        """
        if isinstance(text, str):
            text = [text]
        if isinstance(text_pair, str):
            text_pair = [text_pair]
        return self.batch_encode_plus(text, text_pair=text_pair)

    def _batch_encode_plus(self,
                           text,
                           text_pair=None,
                           return_token_type_ids=None,
                           return_attention_mask=None):
        raise NotImplementedError

    def batch_encode_plus(self,
                          text,
                          text_pair=None,
                          return_token_type_ids=None,
                          return_attention_mask=None):
        return self._batch_encode_plus(text,
                                       text_pair=text_pair,
                                       return_token_type_ids=return_token_type_ids,
                                       return_attention_mask=return_attention_mask)

    def encode(self,
               text,
               text_pair=None):
        res = self.batch_encode_plus(text,
                                     text_pair=text_pair)
        return res["input_ids"]

    @classmethod
    def from_pretrained(cls, name_or_path):
        """
        Arguments:
            model_or_path(str): The path to the model or the directory

                Supports:
                    1. The name_or_path contains the config
                    2. The model_of_path is the output of the method saved_pretrained
                    3. The model_or_path specifics the vocab_file, only appiliable to some tokenizers
        Returns:
             The tokenizer of the corresponding tokenizer.

        Examples:
        """
        tokenizer_class, position_args, kwargs_args = cls._prepare_kwargs_for_tokenizer(name_or_path)
        if tokenizer_class:
            return tokenizer_class(*position_args, **kwargs_args)

        # If no tokenizer class found, it will just call PretrainedTokenizer, so the position arguments should be none
        position_args = ()
        return cls(*position_args, **kwargs_args)

    @classmethod
    def _prepare_kwargs_for_tokenizer(cls, name_or_path):
        """Read files from the given name_or_path and returns the parsed arguments"""
        vocab_file_dict = {}
        tokenizer_type = {}
        class_type = None
        kwargs = {}
        for k, name in cls.VOCAB_FILES.items():
            path = os.path.join(name_or_path, name)
            if os.path.isfile(path):
                vocab_file_dict[k] = path
                # update the vocab file dict into kwargs
                if k not in kwargs:
                    kwargs[k] = path

        for item in cls.FILE_LIST:
            path = os.path.join(name_or_path, item)
            if os.path.isfile(path):
                tokenizer_type[item] = json.load(open(path, 'r'))

        from mindformers.models.bert import BertTokenizer
        mapping_names = {"BertTokenizer": BertTokenizer}
        if 'tokenizer_config.json' in tokenizer_type:
            class_type_str = tokenizer_type['tokenizer_config.json'].get('tokenizer_class', None)
            class_type = mapping_names.get(class_type_str, None)
            kwargs = tokenizer_type['tokenizer_config.json']
            # update the vocab path
            kwargs.update(vocab_file_dict)
        position_args = ()

        return class_type, position_args, kwargs

    def save_pretrained(self, output_path):
        """
        Save the tokenizer by writing the tokenizer_config.json, vocab.txt and special_tokens_map.json to the disk.

        Arguments:
            -output_path-(str): The output file directory.
        """
        tokenizer_config_path = os.path.join(output_path, TOKENIZER_CONFIG_NAME)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        # Start to save the kwargs for the tokenizer
        with open(tokenizer_config_path, 'w') as fp:
            json.dump(self.init_kwargs, fp, indent=4)

        self.save_vocabulary(output_path, VOCAB_FILE_NAME)

    def save_vocabulary(self, save_directory, filename_prefix):
        """Save the vocabulary to the specific path with name_prefix"""
        raise NotImplementedError


class PretrainedTokenizer(PretrainedTokenizerBase):
    """Pretrained Tokenizer provides detailed the tokenizer method."""
    def convert_ids_to_tokens(self, input_ids):
        """Convert the ids to tokens using vocab mapping"""

        return self._convert_ids_to_tokens(input_ids)

    def convert_tokens_to_ids(self, input_tokens):
        """Convert the tokens to ids using vocab mapping"""
        return self._convert_tokens_to_ids(input_tokens)

    def convert_tokens_to_string(self, tokens):
        """Convert the tokens to the string"""
        return " ".join(tokens).strip()

    def tokenize(self, text):
        raise NotImplementedError
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1):
        if token_ids_1:
            return token_ids_0 + token_ids_1
        return token_ids_0

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1):
        if token_ids_1:
            return [0] * (len(token_ids_0[0]) + 1) + [1] * (len(token_ids_1[0]) + 1)
        # cls and sep is 1
        return [0] * (len(token_ids_0[0]) + 1 + 1)

    def _batch_encode_plus(self, text,
                           text_pair=None,
                           return_token_type_ids=None,
                           return_attention_mask=None):
        """Input tex"""
        def get_token_ids(text):
            output = []
            for item in text:
                tokens = self.tokenize(item)
                res = self.convert_tokens_to_ids(tokens)
                output.append(res)
            return output
        text_ids = get_token_ids(text)
        text_pair_ids = get_token_ids(text_pair) if text_pair else None
        return self._batch_prepare_for_model(ids=text_ids,
                                             pair_ids=text_pair_ids,
                                             return_token_type_ids=return_token_type_ids,
                                             return_attention_mask=return_attention_mask)

    def _batch_prepare_for_model(self, ids,
                                 pair_ids=None,
                                 add_special_tokens=True,
                                 return_token_type_ids=None,
                                 return_attention_mask=None):
        """Convert the input_ids to the format of model inputs"""
        output_map = dict()
        def process_token_id(ids, par_ids=None):
            sentece_b_type_ids = []
            if par_ids:
                sentece_b_type_ids = [1] * len(par_ids[0])
            return [0] * len(ids[0]) + sentece_b_type_ids

        if add_special_tokens:
            # add cls and sep: [cls] ids [seq] pair_ids
            input_ids_output = self.build_inputs_with_special_tokens(ids, pair_ids)
            type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
            # two 1 are for cls and sep
            attention_mask = [1] * (1 + 1 + len(ids[0]) + (len(pair_ids[0]) if pair_ids else 0))
        else:
            input_ids_output = [ids[0] + pair_ids[0]] if pair_ids else ids
            attention_mask = attention_mask = [1] * (len(ids[0]) + (len(pair_ids[0]) if pair_ids else 0))
            type_ids = process_token_id(ids, pair_ids)

        output_map['input_ids'] = input_ids_output
        if return_token_type_ids or 'token_type_ids' in self.model_inputs:
            output_map['token_type_ids'] = type_ids
        if return_attention_mask or 'attention_mask' in self.model_inputs:
            output_map['attention_mask'] = attention_mask
        return output_map

    def _tokenize(self, text, **kwargs):
        """Converts the text to tokens"""
        raise NotImplementedError
