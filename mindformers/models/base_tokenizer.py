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
"""Base Tokenizer for the pretrained tokenizer"""
import copy
import os
import json
from collections import defaultdict
import yaml

import numpy as np

import mindspore
from mindspore import Tensor

from mindformers.tools import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig

from .build_tokenizer import build_tokenizer

__all__ = ['BaseTokenizer', 'Tokenizer', 'SpecialTokensMixin']

from ..tools.download_tools import downlond_with_progress_bar
from ..mindformer_book import MindFormerBook, print_path_or_list

SPECIAL_TOKEN_FILE_NAME = 'special_tokens_map.json'
TOKENIZER_CONFIG_NAME = 'tokenizer_config.json'


class SpecialTokensMixin:
    """A class for managing the specific tokens"""
    SPECIAL_TOKENS = ['pad_token', 'cls_token', 'sep_token', 'unk_token', 'mask_token', 'bos_token', 'eos_token']

    def __init__(self,
                 **kwargs):
        self._pad_token = None
        self._sep_token = None
        self._cls_token = None
        self._mask_token = None
        self._bos_token = None
        self._eos_token = None
        self._pad_token_type_id = 0
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

    @property
    def eos_token(self):
        return self._eos_token

    @property
    def eos_token_id(self):
        return self._convert_tokens_to_ids(self._eos_token)

    @property
    def bos_token(self):
        return self._bos_token

    @property
    def bos_token_id(self):
        return self._convert_tokens_to_ids(self._bos_token)

    @property
    def pad_token_type_id(self):
        return self._pad_token_type_id

    @property
    def all_specifical_token_index(self):
        all_specifical_token_index = []
        for item in self.SPECIAL_TOKENS:
            if hasattr(self, '_' + item):
                cur_item = getattr(self, '_' + item)
                if cur_item:
                    all_specifical_token_index.append(self._convert_tokens_to_ids(cur_item))
        return all_specifical_token_index



class BaseTokenizer(SpecialTokensMixin):
    """The pretrained tokenize providing basic method for tokenizing."""
    MODEL_INPUT_NAME = ['input_ids', 'attention_mask', 'token_type_ids']
    VOCAB_FILES = {}
    FILE_LIST = []

    def __init__(self, **kwargs):
        super(BaseTokenizer, self).__init__(**kwargs)
        self.model_inputs = self.MODEL_INPUT_NAME
        self.init_kwargs = kwargs

    def __call__(self,
                 text,
                 text_pair=None,
                 add_special_tokens=True,
                 max_length=None,
                 padding=False,
                 return_tensors=None,
                 **kwargs):
        """
        Get the text inputs and then return the padded ids

        It needs three steps:
            1. tokenize
            2. convert them to ids
            3. combine them to a batch
        """
        return_batch = True
        if isinstance(text, str):
            return_batch = False
        output = self.batch_encode_plus(text, text_pair=text_pair, max_length=max_length,
                                        add_special_tokens=add_special_tokens,
                                        padding=padding,
                                        return_tensors=return_tensors,
                                        return_batch=return_batch,
                                        **kwargs)
        return output

    def _batch_encode_plus(self,
                           text,
                           text_pair=None,
                           max_length=None,
                           padding_strategy="do_not_pad",
                           add_special_tokens=True,
                           return_tensors=None,
                           return_token_type_ids=None,
                           return_attention_mask=None,
                           return_batch=True,
                           **kwargs):
        """Convert the input text into the ids"""
        raise NotImplementedError

    def batch_encode_plus(self,
                          text,
                          text_pair=None,
                          max_length=None,
                          padding=None,
                          add_special_tokens=True,
                          return_token_type_ids=None,
                          return_attention_mask=None,
                          return_tensors=None,
                          return_batch=True,
                          **kwargs):
        """
        Convert the input text into the list. This API can process the batch inputs.
        """
        if padding and padding != "max_length":
            raise ValueError("padding only supports `max_length` or `None`.")
        padding_strategy = None
        if padding:
            padding_strategy = "max_length"
        if max_length and not padding:
            logger.warning("If you want to enable the padding, please set padding to `max_length`.")
        # if input text is only one list, we should prepare it into a tensor with batch size 1.
        text = self._prepare_input_to_list(text)
        text_pair = self._prepare_input_to_list(text_pair)
        return self._batch_encode_plus(text,
                                       text_pair=text_pair,
                                       max_length=max_length,
                                       padding_strategy=padding_strategy,
                                       add_special_tokens=add_special_tokens,
                                       return_tensors=return_tensors,
                                       return_token_type_ids=return_token_type_ids,
                                       return_attention_mask=return_attention_mask,
                                       return_batch=return_batch,
                                       **kwargs)

    def _prepare_input_to_list(self, inputs):
        """put the input into the list"""
        if inputs is None:
            return inputs
        if not isinstance(inputs, list):
            inputs = [inputs]
        return inputs

    def _get_token_ids(self, text):
        """Get the token_ids"""
        if not isinstance(text, list):
            tokens = self.tokenize(text)
            res = self.convert_tokens_to_ids(tokens)
            return res
        output = []
        for item in text:
            tokens = self.tokenize(item)
            res = self.convert_tokens_to_ids(tokens)
            output.append(res)
        return output

    def encode(self,
               text,
               text_pair=None,
               max_length=None,
               padding=None,
               return_tensors=None,
               add_special_tokens=True,
               return_batch=False):
        """Converts the text to the processed ids"""
        res = self.batch_encode_plus(text=text,
                                     text_pair=text_pair,
                                     padding=padding,
                                     max_length=max_length,
                                     return_tensors=return_tensors,
                                     add_special_tokens=add_special_tokens,
                                     return_batch=return_batch)
        return res["input_ids"]

    @classmethod
    def from_pretrained(cls, name_or_path):
        """
        Arguments:
            name_or_path(str): The path to the model or the directory

                Supports:
                    1. The name_or_path contains the config
                    2. The model_of_path is the output of the method saved_pretrained
                    3. The model_or_path specifics the vocab_file, only applicable to some tokenizers
        Returns:
             The tokenizer of the corresponding tokenizer.

        Examples:
        """
        kwargs = dict()
        class_name = None
        loaded_kwargs = {}
        if name_or_path in MindFormerBook.get_tokenizer_support_list():
            config, cache_path = cls._download_using_name(name_or_path)
            class_name, loaded_kwargs = cls._get_class_name_and_args_form_config(config)
            logger.info("Download the tokenizer finished, modify the input name_or_path "
                        "from %s to %s.", name_or_path, cache_path)
            name_or_path = cache_path

        yaml_list = None
        if os.path.isdir(name_or_path):
            yaml_list = [file for file in os.listdir(name_or_path) if file.endswith(".yaml")]
            if len(yaml_list) > 1:
                logger.warning("There should be only one yaml file under the directory %s, "
                               "but followings are found: %s", name_or_path, yaml_list)
        if yaml_list:
            yaml_file = os.path.join(name_or_path, yaml_list[0])
            logger.info("config in the yaml file %s are used for tokenizer building.", yaml_file)
            config = MindFormerConfig(yaml_file)
            class_name, loaded_kwargs = cls._get_class_name_and_args_form_config(config)

        vocab_dict, file_dict = cls.read_files_according_specific_by_tokenizer(name_or_path)
        if 'tokenizer_config.json' in file_dict:
            class_name = file_dict['tokenizer_config.json'].pop('tokenizer_class', None)
            loaded_kwargs = file_dict['tokenizer_config.json']
        else:
            logger.warning("Can't find the tokenizer_config.json in the file_dict. "
                           "The content of file_dict is : %s", file_dict)
        kwargs.update(loaded_kwargs)
        kwargs.update(vocab_dict)
        if not class_name:
            class_name = cls.__name__
        logger.info("build tokenizer class name is: %s using args %s.", class_name, kwargs)
        return build_tokenizer(class_name=class_name, **kwargs)

    @classmethod
    def _download_using_name(cls, name_or_path):
        """Given the supported model name, download it from the urls"""
        cache_path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                  name_or_path.split("_")[0])
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        yaml_file = os.path.join(cache_path, name_or_path + ".yaml")
        if not os.path.exists(yaml_file):
            url = MindFormerBook.get_model_config_url_list()[name_or_path][0]
            logger.info("Download from the url %s to %s", url, yaml_file)
            downlond_with_progress_bar(url, yaml_file)

        url_vocab = MindFormerBook.get_tokenizer_support_list()[name_or_path][0]
        local_vocab_name = url_vocab.split('/')[-1]
        vocab_file = os.path.join(cache_path, local_vocab_name)
        if not os.path.exists(vocab_file):
            logger.info("Download the yaml from the url %s to %s.", url_vocab, vocab_file)
            downlond_with_progress_bar(url_vocab, vocab_file)
        config = MindFormerConfig(yaml_file)
        return config, cache_path

    @classmethod
    def cache_vocab_files(cls, name_or_path, cache_path=None):
        """Cache the vocab files to the default dir"""
        if not cache_path:
            cache_path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                      name_or_path.split("_")[0])
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
        url_vocab = MindFormerBook.get_tokenizer_support_list()[name_or_path][0]
        local_vocab_name = url_vocab.split('/')[-1]
        vocab_file = os.path.join(cache_path, local_vocab_name)
        if not os.path.exists(vocab_file):
            logger.info("Download the vocab file from the url %s to %s.", url_vocab, vocab_file)
            downlond_with_progress_bar(url_vocab, vocab_file)
        return vocab_file

    @classmethod
    def _get_class_name_and_args_form_config(cls, config):
        """Lookup the yaml files under the name_or_path"""
        class_name = None
        tokenizer_args = {}
        if config and 'processor' in config and 'tokenizer' in config['processor'] \
                and config.processor.tokenizer and'type' in config.processor.tokenizer:
            tokenizer_args = config['processor']['tokenizer']
            class_name = tokenizer_args.pop('type', None)
            logger.info("Read the tokenizer name %s from %s. The load kwargs for tokenizer "
                        "is: %s", class_name, config, tokenizer_args)
        else:
            logger.info("There is no matched format config['processor']['tokenizer']  in config %s", config)
        return class_name, tokenizer_args

    @classmethod
    def read_files_according_specific_by_tokenizer(cls, name_or_path):
        """Read the file path specific by the class variable in the tokenizer"""
        read_vocab_file_dict = {}
        read_tokenizer_file_dict = {}
        for k, name in cls.VOCAB_FILES.items():
            if isinstance(name, str):
                path = os.path.join(name_or_path, name)
                if os.path.isfile(path):
                    read_vocab_file_dict[k] = path
            # To support tokenizer like clip that has two types for vocab files.
            elif isinstance(name, list):
                for sub_name in name:
                    path = os.path.join(name_or_path, sub_name)
                    if os.path.isfile(path):
                        read_vocab_file_dict[k] = path

        for item in cls.FILE_LIST:
            path = os.path.join(name_or_path, item)
            if os.path.isfile(path):
                read_tokenizer_file_dict[item] = json.load(open(path, 'r'))
        logger.info("Tokenizer %s read tokenizer files from %s are:"
                    "%s and %s", cls.__name__, name_or_path, read_vocab_file_dict, read_tokenizer_file_dict)
        return read_vocab_file_dict, read_tokenizer_file_dict

    def _pad(self, id_dict, max_length, padding_strategy="do_not_pad"):
        """Do padding according to the max_length"""
        if not max_length or padding_strategy != "max_length":
            return id_dict
        is_batch = False
        if isinstance(id_dict['input_ids'], list) and isinstance(id_dict['input_ids'][0], list):
            is_batch = True

        def _pad_batch(source_ids, pad_value):
            if not is_batch:
                source_ids = [source_ids]
            for i in range(len(source_ids)):
                if max_length < len(source_ids[i]):
                    raise ValueError(f"The length of input_ids {len(source_ids[i])} "
                                     f"exceeds the max_length {max_length}, "
                                     f"please increase the max_length.")
                source_ids[i] += [pad_value] * (max_length - len(source_ids[i]))
            if not is_batch:
                source_ids = source_ids[0]

        _pad_batch(id_dict['input_ids'], pad_value=self.pad_token_id)
        if "attention_mask" in id_dict:
            _pad_batch(id_dict['attention_mask'], pad_value=0)
        if "token_type_ids" in id_dict:
            _pad_batch(id_dict['token_type_ids'], pad_value=self.pad_token_type_id)
        return id_dict

    def prepare_for_model(self,
                          ids,
                          pair_ids=None,
                          add_special_tokens=True,
                          max_length=None,
                          padding_strategy="do_not_pad",
                          return_tensors=None,
                          return_token_type_ids=None,
                          return_attention_mask=None):
        """
        Insert the special ids into the input ids, generate the attention mask and do padding.
        """
        if not isinstance(ids, list):
            raise ValueError("The input ids should be a list.")
        output_map = dict()

        def process_token_id(ids, par_ids=None):
            sentence_b_type_ids = []
            if par_ids:
                sentence_b_type_ids = [1] * len(par_ids[0])
            return [0] * len(ids[0]) + sentence_b_type_ids

        if add_special_tokens:
            # add cls and sep: [cls] ids [seq] pair_ids
            input_ids_output = self.build_inputs_with_special_tokens(ids, pair_ids)
            type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
            # two 1 are for cls and sep
            attention_mask = [1] * (1 + 1 + len(ids) + (len(pair_ids) if pair_ids else 0))
        else:
            input_ids_output = [ids + pair_ids] if pair_ids else ids
            attention_mask = attention_mask = [1] * (len(ids) + (len(pair_ids) if pair_ids else 0))
            type_ids = process_token_id(ids, pair_ids)

        output_map['input_ids'] = input_ids_output
        if return_token_type_ids or 'token_type_ids' in self.model_inputs:
            output_map['token_type_ids'] = type_ids
        if return_attention_mask or 'attention_mask' in self.model_inputs:
            output_map['attention_mask'] = attention_mask

        output_map = self._pad(output_map, max_length=max_length, padding_strategy=padding_strategy)
        if return_tensors and return_tensors != 'ms':
            raise ValueError("You should set return_tensors to be `ms`.")
        if return_tensors:
            for k, v in output_map.items():
                output_map[k] = Tensor(v)
        return output_map

    def save_pretrained(self, save_directory=None, save_name="mindspore_model", file_format='yaml'):
        """
        Save the tokenizer by writing the tokenizer_config.json, vocab.txt and special_tokens_map.json to the disk.

        Arguments:
            save_directory(str): The output file directory.
            save_name(str):
            file_format(str): Support json or yaml.
        """
        default_directory = MindFormerBook.get_default_checkpoint_save_folder()
        if save_directory is None and not os.path.exists(default_directory):
            save_directory = default_directory
            os.makedirs(save_directory)
        if file_format not in ('yaml', 'json'):
            raise ValueError(f"format should be one of [`yaml`, `json`], but got {file_format}.")

        kwargs = copy.deepcopy(self.init_kwargs)
        # Start to save the kwargs for the tokenizer
        if file_format == 'yaml':
            kwargs['type'] = self.__class__.__name__
            yaml_list = [file for file in os.listdir(save_directory) if file.endswith(".yaml")]
            merged_dict = dict()
            if len(yaml_list) > 1:
                raise ValueError(f"There should be only one yaml file under the directory {save_directory}.")
            if not yaml_list:
                logger.info("The yaml is not found under the %s, so create a new one. "
                            "Start to create a new one.", save_directory)
                yaml_file = os.path.join(save_directory, save_name + '.yaml')
            else:
                yaml_file = os.path.join(save_directory, yaml_list[0])
                with open(yaml_file, 'r') as file_reader:
                    merged_dict = yaml.load(file_reader.read(), Loader=yaml.Loader)
            logger.info("Dumping tokenizer args to %s.", yaml_file)
            if 'processor' not in merged_dict:
                merged_dict['processor'] = dict()
            merged_dict['processor']['tokenizer'] = kwargs
            with open(yaml_file, 'w') as file_reader:
                yaml.dump(merged_dict, file_reader)
        else:
            kwargs["tokenizer_class"] = self.__class__.__name__
            tokenizer_config_path = os.path.join(save_directory, TOKENIZER_CONFIG_NAME)
            with open(tokenizer_config_path, 'w') as fp:
                json.dump(kwargs, fp, indent=4)

        output_name = self.VOCAB_FILES['vocab_file']
        if isinstance(output_name, list):
            output_name = output_name[0]
        self.save_vocabulary(save_directory, output_name)

    def save_vocabulary(self, save_directory, filename_prefix):
        """Save the vocabulary to the specific path with name_prefix"""
        raise NotImplementedError

    def decode(self,
               token_ids,
               skip_special_tokens=False,
               **kwargs):
        """Converts the token_ids to the string"""
        if isinstance(token_ids, mindspore.Tensor):
            token_ids = token_ids.asnumpy().tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        if not isinstance(token_ids, list):
            raise TypeError(f"`token_ids` should be the list, but got {type(token_ids)}.")
        output = self._decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        return output

    def _decode(self, token_ids,
                skip_special_tokens,
                **kwargs):
        """The basic function of the decode"""
        raise NotImplementedError


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class Tokenizer(BaseTokenizer):
    """Pretrained Tokenizer provides detailed the tokenizer method."""
    _support_list = []
    def convert_ids_to_tokens(self, input_ids, skip_special_tokens=False):
        """Convert the ids to tokens using vocab mapping"""
        output = []
        for item in input_ids:
            if skip_special_tokens and item in self.all_specifical_token_index:
                continue
            else:
                output.append(self._convert_ids_to_tokens(item))
        return output

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
            return [0] * (len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 1)
        # cls and sep is 1
        return [0] * (len(token_ids_0) + 1 + 1)

    def _batch_encode_plus(self, text,
                           text_pair=None,
                           max_length=None,
                           padding_strategy="do_not_pad",
                           add_special_tokens=True,
                           return_tensors=None,
                           return_token_type_ids=None,
                           return_attention_mask=None,
                           return_batch=True,
                           **kwargs):
        """Convert the text into the converted id. text should be batched. For example, [["hello world"]]"""
        if not isinstance(text, list) and not isinstance(text[0], list):
            raise ValueError("For _batch_encode_plus, the input `text` should be batched, "
                             "for example: [['hello world']].")

        text_ids = [self._get_token_ids(item) for item in text]
        text_pair_ids = [self._get_token_ids(item) for item in text_pair] if text_pair else None
        processed_output = self._batch_prepare_for_model(ids=text_ids,
                                                         pair_ids=text_pair_ids,
                                                         max_length=max_length,
                                                         padding_strategy=padding_strategy,
                                                         add_special_tokens=add_special_tokens,
                                                         return_tensors=return_tensors,
                                                         return_token_type_ids=return_token_type_ids,
                                                         return_attention_mask=return_attention_mask,
                                                         return_batch=return_batch)
        return processed_output

    def _batch_prepare_for_model(self, ids,
                                 pair_ids=None,
                                 add_special_tokens=True,
                                 max_length=None,
                                 padding_strategy="do_not_pad",
                                 return_tensors=None,
                                 return_token_type_ids=None,
                                 return_attention_mask=None,
                                 return_batch=True):
        """Convert the input_ids to the format of model inputs"""
        if return_tensors and return_tensors != 'ms':
            raise ValueError("You should set return_tensors to be `ms`.")
        if not return_batch and len(ids) != 1:
            raise ValueError(f"If `return_batch` is True, the length of input ids should be 1. But found {len(ids)}."
                             f"Input ids is: {ids}")
        if pair_ids:
            paired_ids = zip(ids, pair_ids)
        else:
            paired_ids = zip(ids, [None] * len(ids))
        output = defaultdict(list)
        for per_ids, per_pair_ids in paired_ids:
            per_output = self.prepare_for_model(ids=per_ids,
                                                pair_ids=per_pair_ids,
                                                add_special_tokens=add_special_tokens,
                                                max_length=None,
                                                padding_strategy="do_not_pad",
                                                return_tensors=None,
                                                return_token_type_ids=return_token_type_ids,
                                                return_attention_mask=return_attention_mask)
            if not return_batch:
                output = per_output
            else:
                for k, v in per_output.items():
                    output[k].append(v)
        output_map = self._pad(output, max_length=max_length, padding_strategy=padding_strategy)
        if return_tensors:
            for k in output_map.keys():
                output_map[k] = Tensor(output_map[k])
        return output_map

    def _decode(self, token_ids, skip_special_tokens=False, **kwargs):
        ids = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        return self.convert_tokens_to_string(ids)


    def _tokenize(self, text, **kwargs):
        """Converts the text to tokens"""
        raise NotImplementedError

    @property
    def vocab_size(self):
        """Get the vocab size of the """
        raise NotImplementedError

    @classmethod
    def show_support_list(cls):
        """show_support_list method"""
        logger.info("support list of %s is:", cls.__name__)
        print_path_or_list(cls._support_list)

    @classmethod
    def get_support_list(cls):
        """get_support_list method"""
        return cls._support_list
