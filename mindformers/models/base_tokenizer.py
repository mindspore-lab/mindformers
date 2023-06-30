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
# This file was refer to project:
# https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils.py
# ============================================================================
"""Base Tokenizer for the pretrained tokenizer"""
from typing import Optional, List, Union

import copy
import os
import json
import shutil
from collections import defaultdict
import yaml

import numpy as np

import mindspore
from mindspore import Tensor

from mindformers.tools import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig

from .build_tokenizer import build_tokenizer

__all__ = ['BaseTokenizer', 'Tokenizer', 'SpecialTokensMixin']

from ..tools.download_tools import download_with_progress_bar
from ..tools.utils import try_sync_file
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
        self._unk_token = None
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
    def unk_token(self):
        return self._unk_token

    @property
    def unk_token_id(self):
        return self._convert_tokens_to_ids(self._unk_token)

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
    _model_type = 0
    _model_name = 1

    def __init__(self, **kwargs):
        super(BaseTokenizer, self).__init__(**kwargs)
        self.model_inputs = self.MODEL_INPUT_NAME
        self.init_kwargs = kwargs

    def __call__(self,
                 text: Optional[Union[str, List[str]]],
                 text_pair: Optional[Union[str, List[str]]] = None,
                 add_special_tokens: bool = True,
                 max_length: Optional[int] = None,
                 padding: str = False,
                 truncation: bool = False,
                 return_tensors: Optional[bool] = None,
                 **kwargs):
        r"""
        Tokenize the input string and convert them into the ids.

        Args:
            text(str, list(str)) : To be converted text strings. It can be string or a list of strings.
            text_pair(str, list(str)): To be converted text pair strings. It can be string or a list of strings.
            add_special_tokens(bool): Whether to add special tokens such as CLS and EOS to the token list. The subclass
                can determine the behavior of the adding by overriding the method `build_inputs_with_special_tokens`.
                If True, the special token will be added. Default True.
            max_length (int): max length of tokenizer's output . Default None.
            padding(False / "max_length"): padding for max_length. Default None.
            truncation(bool): To truncate the sequence if the length exceeds the max_length. Default False.
            return_tensors(str): Specific the returned tensor type. If set None, the returned tensor will be
                `numpy.ndarray`. If set `ms`, the returned tensor will be of `mindspore.Tensor`.
                Support 'ms' and None. Default None.
            **kwargs: The other kwargs passed to the internal behaivor, currently not used.

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
        return_batch = True
        if isinstance(text, str):
            return_batch = False
        output = self.batch_encode_plus(text, text_pair=text_pair, max_length=max_length,
                                        add_special_tokens=add_special_tokens,
                                        padding=padding,
                                        truncation=truncation,
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
                           truncation=False,
                           return_tensors=None,
                           return_token_type_ids=None,
                           return_attention_mask=None,
                           return_batch=True,
                           **kwargs):
        """Convert the input text into the ids"""
        raise NotImplementedError

    def truncate_sequences(self, ids, id_pairs, nums_tokens_to_remove):
        if nums_tokens_to_remove <= 0:
            return ids, id_pairs
        if id_pairs:
            raise ValueError("The id_pairs do not support truncation, please set it to be a empty list or None.")

        ids = ids[:-nums_tokens_to_remove]
        return ids, id_pairs

    def batch_encode_plus(self,
                          text: Optional[Union[str, List[str]]],
                          text_pair: Optional[Union[str, List[str]]] = None,
                          max_length: Optional[int] = None,
                          padding: Optional[str] = None,
                          add_special_tokens: bool = True,
                          truncation: bool = False,
                          return_token_type_ids: Optional[bool] = None,
                          return_attention_mask: Optional[bool] = None,
                          return_tensors: Optional[bool] = None,
                          return_batch: bool = True,
                          **kwargs):
        r"""
        The core function of the __call__ method. The method aims to tokenizer the input strings and then convert them
        into ids.

        Args:
            text(str) : To be converted text strings. It can be string or a list of strings.
            text_pair(str): To be converted text pair strings. It can be string or a list of strings.
            max_length (int): max length of tokenizers output . Default None.
            padding(bool, str): padding for max_length. Default None.
            truncation(bool): To truncate the sequence if the length exceeds the max_length. Default False.
            add_special_tokens(bool): Whether to add special tokens such as CLS and EOS to the token list. The subclass
                can determine the behavior of the adding by overriding the method `build_inputs_with_special_tokens`.
                If True, the special token will be added. Default True.
            return_token_type_ids(bool): Whether to add `token_type_ids` in the returned dict. If True,
                `token_type_ids` will be added, otherwise not. If None, it will be added if it is in the
                MODEL_INPUT_NAME. Default None.
            return_attention_mask(bool): Whether to add `return_attention_mask` in the returned dict. If True,
                `return_attention_mask` will be added, otherwise not. If None, it will be added if it is in the
                MODEL_INPUT_NAME. Default None.
            return_tensors(str): Specific the returned tensor type. If support `ms` only, the returned value in the
                dict will be converted to the mindspore.Tensor, otherwise it will return the list.
                Default None.
            return_batch(bool): Whether the returned the list should be added batch dimension. Default True.
            **kwargs: The other kwargs passed to the internal method, currently not used.

        Outputs:
            A dict contains the processed ids, attention_mask that specific by the member `MODEL_INPUT_NAME`
            of the subclass.
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
                                       truncation=truncation,
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
        if len(text) == 1 and isinstance(text[0], str) and output and isinstance(output[0], list):
            output = output[0]
        return output

    def encode(self,
               text: Optional[Union[str, List[str]]],
               text_pair: Optional[Union[str, List[str]]] = None,
               max_length: Optional[int] = None,
               padding: Optional[str] = None,
               return_tensors: Optional[bool] = None,
               add_special_tokens: bool = True,
               return_batch: bool = False):
        """
        Convert the input strings to the id list. The arguments are mostly same with the `batch_encode_plus`, but
        it returns the token id list rather than the python dict.

        Examples:
            >>> from mindformers import T5Tokenizer
            >>> tokenizer = T5Tokenizer.from_pretrained("t5_small")
            >>> res = tokenizer.encode("hello world!")
            >>> print(res)
            [21820, 296, 55, 1]

        Returns:
            A list of token ids mapped by the vocabulary.
        """
        res = self.batch_encode_plus(text=text,
                                     text_pair=text_pair,
                                     padding=padding,
                                     max_length=max_length,
                                     return_tensors=return_tensors,
                                     add_special_tokens=add_special_tokens,
                                     return_batch=return_batch)
        return res["input_ids"]

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs):
        """
        Instantiates a tokenizer by the name_or_path. User can get the name using `get_support_list` of any tokenizer,
        it will download the necessary files from the cloud. or pass a directory where contains the vocabulary file
        and tokenizers yaml configuration file.

        Args:
            name_or_path (str): It supports the following two input types: If the name_or_path is a supported tokenizer
                name, for example, `clip_vit_b_32` and `t5_small`, it will download the necessary files from the cloud.
                User can select one from the support list by call `MindFormerBook.show_tokenizer_support_list()`.
                If name_or_path is a path to the local directory where there should have vocaburary files and
                configuration file ended with `yaml`. The vocaburary file needed by the tokenizer is determined
                by `.VOCAB_FILES`.
            pretrained_model_name_or_path (Optional[str]): Equal to "name_or_path",
                if "pretrained_model_name_or_path" is set, "name_or_path" is useless.

        Examples:
            >>> from mindformers import T5Tokenizer
            >>> tokenizer = T5Tokenizer.from_pretrained("t5_small")
            >>> res = tokenizer.encode("hello world!")
            >>> print(res)
            [21820, 296, 55, 1]

        Returns:
            A instanced tokenizer.
        """
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        if pretrained_model_name_or_path is not None:
            name_or_path = pretrained_model_name_or_path

        is_exist = os.path.exists(name_or_path)
        is_dir = os.path.isdir(name_or_path)
        if not is_exist and (name_or_path not in cls._support_list):
            raise ValueError(f'{name_or_path} does not exist,'
                             f' or it is not supported by {cls.__name__}. '
                             f'please select from {cls._support_list}.')

        if is_exist and not is_dir:
            raise ValueError(f"{name_or_path} is not a directory.")

        kwargs = dict()
        class_name = None
        loaded_kwargs = {}
        if name_or_path in MindFormerBook.get_tokenizer_url_support_list():
            config, cache_path = cls._download_using_name(name_or_path)
            class_name, loaded_kwargs = cls._get_class_name_and_args_form_config(config)
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
        tokenizer_name = name_or_path
        if name_or_path.startswith('mindspore'):
            # Adaptation the name of tokenizer at the beginning of mindspore,
            # the relevant file will be downloaded from the Xihe platform.
            # such as "mindspore/clip_vit_b_32"
            tokenizer_name = name_or_path.split('/')[cls._model_name]
            cache_path = os.path.join(MindFormerBook.get_xihe_checkpoint_download_folder(),
                                      tokenizer_name.split('_')[cls._model_type])
        else:
            # Default the name of tokenizer,
            # the relevant file will be downloaded from the Obs platform.
            # such as "clip_vit_b_32"
            cache_path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                      name_or_path.split('_')[cls._model_type])

        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)

        yaml_file = os.path.join(cache_path, tokenizer_name + ".yaml")

        def get_default_yaml_file(model_name):
            default_yaml_file = ""
            for model_dict in MindFormerBook.get_trainer_support_task_list().values():
                if model_name in model_dict:
                    default_yaml_file = model_dict.get(model_name)
                    break
            return default_yaml_file

        if not os.path.exists(yaml_file):
            default_yaml_file = get_default_yaml_file(tokenizer_name)
            if os.path.realpath(default_yaml_file) and os.path.exists(default_yaml_file):
                shutil.copy(default_yaml_file, yaml_file)
                logger.info("default yaml config in %s is used.", yaml_file)
            else:
                raise FileNotFoundError(f'default yaml file path must be correct, but get {default_yaml_file}')

        # some tokenizers rely on more than one file, e.g gpt2
        tokenizer_need_files = MindFormerBook.get_tokenizer_url_support_list()[name_or_path]
        for url_file in tokenizer_need_files:
            local_file_name = url_file.split('/')[-1]
            file_path = os.path.join(cache_path, local_file_name)
            if not os.path.exists(file_path):
                logger.info("Download the vocab from the url %s to %s.", url_file, file_path)
                download_with_progress_bar(url_file, file_path)
            try_sync_file(file_path)

        config = MindFormerConfig(yaml_file)
        return config, cache_path

    @classmethod
    def cache_vocab_files(cls, name_or_path, cache_path=None):
        """Cache the vocab files to the default dir"""
        if not cache_path:
            cache_path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                      name_or_path.split("_")[cls._model_type])
            if not os.path.exists(cache_path):
                os.makedirs(cache_path, exist_ok=True)

        # some tokenizers rely on more than one file, e.g gpt2
        tokenizer_need_files = MindFormerBook.get_tokenizer_url_support_list()[name_or_path]
        for url_file in tokenizer_need_files:
            local_file_name = url_file.split('/')[-1]
            file_path = os.path.join(cache_path, local_file_name)
            if not os.path.exists(file_path):
                logger.info("Download the yaml from the url %s to %s.", url_file, file_path)
                download_with_progress_bar(url_file, file_path)
            try_sync_file(file_path)
        read_vocab_file_dict, _ = cls.read_files_according_specific_by_tokenizer(cache_path)
        return read_vocab_file_dict

    @classmethod
    def _get_class_name_and_args_form_config(cls, config):
        """Lookup the yaml files under the name_or_path"""
        class_name = None
        tokenizer_args = {}
        if config and 'processor' in config and 'tokenizer' in config['processor'] \
                and config.processor.tokenizer and'type' in config.processor.tokenizer:
            tokenizer_args = config['processor']['tokenizer']
            class_name = tokenizer_args.pop('type', None)
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
        return read_vocab_file_dict, read_tokenizer_file_dict

    def _pad(self, id_dict, max_length, padding_strategy="do_not_pad", return_attention_mask=None):
        """Do padding according to the max_length"""
        is_batch = False
        if isinstance(id_dict['input_ids'], list) and isinstance(id_dict['input_ids'][0], list):
            is_batch = True
            length_each = [len(line) for line in id_dict['input_ids']]
            for item in length_each:
                if length_each[0] != item and (not max_length or padding_strategy != "max_length"):
                    raise ValueError(f"You should set `max_length` to {max(length_each)} "
                                     f"and padding_strategy to `max_length`, as the length in the batch "
                                     f"is different, which should be padded.")


        if return_attention_mask is not False:
            return_attention_mask = True

        if return_attention_mask and 'attention_mask' in self.model_inputs:
            if is_batch:
                id_dict['attention_mask'] = [[1] * len(line) for line in id_dict['input_ids']]
            else:
                id_dict['attention_mask'] = [1] * len(id_dict['input_ids'])

        if not max_length or padding_strategy != "max_length":
            return id_dict

        def _pad_batch(source_ids, pad_value):
            if not is_batch:
                source_ids = [source_ids]
            for i in range(len(source_ids)):
                if max_length < len(source_ids[i]):
                    raise ValueError(f"The length of input_ids {len(source_ids[i])} "
                                     f"exceeds the max_length {max_length}, "
                                     f"please increase the `max_length` of the tokenizer.")
                source_ids[i] += [pad_value] * (max_length - len(source_ids[i]))
            if not is_batch:
                source_ids = source_ids[0]

        _pad_batch(id_dict['input_ids'], pad_value=self.pad_token_id)
        if "attention_mask" in id_dict:
            _pad_batch(id_dict['attention_mask'], pad_value=0)
        if "token_type_ids" in id_dict:
            _pad_batch(id_dict['token_type_ids'], pad_value=self.pad_token_type_id)

        return id_dict

    def postprocess_ids(self,
                        ids,
                        pair_ids=None,
                        add_special_tokens=True,
                        max_length=None,
                        truncation=False,
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
                sentence_b_type_ids = [1] * len(par_ids)
            return [0] * len(ids) + sentence_b_type_ids

        length_of_each = len(ids)

        if max_length and truncation:
            num_sp_tokens = self.num_special_tokens_to_add() if add_special_tokens else 0
            num_tokens = length_of_each + num_sp_tokens - max_length
            ids, pair_ids = self.truncate_sequences(ids, pair_ids, num_tokens)

        if add_special_tokens:
            # add cls and sep: [cls] ids [seq] pair_ids
            input_ids_output = self.build_inputs_with_special_tokens(ids, pair_ids)
            type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            input_ids_output = [ids + pair_ids] if pair_ids else ids
            type_ids = process_token_id(ids, pair_ids)

        output_map['input_ids'] = input_ids_output
        if return_token_type_ids or 'token_type_ids' in self.model_inputs:
            output_map['token_type_ids'] = type_ids

        output_map = self._pad(output_map, max_length=max_length, padding_strategy=padding_strategy,
                               return_attention_mask=return_attention_mask)

        if return_tensors and return_tensors != 'ms':
            raise ValueError("You should set return_tensors to be `ms`.")
        if return_tensors:
            for k, v in output_map.items():
                v = np.array(v)
                if v.dtype == np.int64:
                    v = v.astype(np.int32)
                output_map[k] = Tensor(v)
        return output_map

    def save_pretrained(self,
                        save_directory: Optional[str] = None,
                        save_name: str = "mindspore_model",
                        file_format: str = 'yaml'):
        """
        Save the tokenizer by writing the `save_name`.yaml and vocaburary files those are determinied by `.VOCAB_FILES`
        to the disk. The kwargs passed to initialize the tokenizer will be saved.

        Args:
            save_directory(str): The output file directory. If None, the directory will be  `./checkpoint_save`,
                which can be obtained by the `MindFormerBook.get_default_checkpoint_save_folder()`. Default None.
            save_name(str): The file name of the saved files. Default mindspore_model.
            file_format(str): Support json or yaml. Default yaml.

        Examples:
            >>> from mindformers import T5Tokenizer, MindFormerBook
            >>> tokenizer = T5Tokenizer.from_pretrained("t5_small")
            >>> tokenizer.save_pretrained()
            >>> output_path = MindFormerBook.get_default_checkpoint_save_folder()
            >>> print(os.listdir(output_path))
            ['mindspore_model.yaml', 'spiece.model']

        """
        default_directory = MindFormerBook.get_default_checkpoint_save_folder()
        if save_directory is None:
            save_directory = default_directory
            os.makedirs(save_directory, exist_ok=True)
        if save_name is None:
            save_name = "mindspore_model"
        if file_format not in ('yaml', 'json'):
            raise ValueError(f"format should be one of [`yaml`, `json`], but got {file_format}.")

        kwargs = copy.deepcopy(self.init_kwargs)
        # Start to save the kwargs for the tokenizer
        if file_format == 'yaml':
            kwargs['type'] = self.__class__.__name__
            merged_dict = dict()

            yaml_file = os.path.join(save_directory, save_name + '.yaml')
            if os.path.exists(yaml_file):
                with open(yaml_file, 'r') as file_reader:
                    merged_dict = yaml.load(file_reader.read(), Loader=yaml.Loader)
                    if merged_dict is None:
                        merged_dict = dict()

            processor_name = MindFormerBook.get_tokenizer_name_to_processor()[kwargs['type']]
            merged_dict['processor'] = {"type": processor_name}
            merged_dict['processor']['tokenizer'] = kwargs
            with open(yaml_file, 'w') as file_reader:
                yaml.dump(merged_dict, file_reader)
        elif file_format == 'json':
            kwargs["tokenizer_class"] = self.__class__.__name__
            tokenizer_config_path = os.path.join(save_directory, TOKENIZER_CONFIG_NAME)
            with open(tokenizer_config_path, 'w') as fp:
                json.dump(kwargs, fp, indent=4)
        else:
            raise ValueError(f"file_format should be one of [json, yaml], but got {file_format}.")

        # some tokenizers rely on more than one file, e.g gpt2
        name_keys = self.VOCAB_FILES.keys()
        for name_key in name_keys:
            output_name = self.VOCAB_FILES[name_key]
            if isinstance(output_name, list):
                output_name = output_name[0]
            self.save_vocabulary(save_directory, output_name)

    def save_vocabulary(self, save_directory, filename_prefix):
        """Save the vocabulary to the specific path with name_prefix"""
        raise NotImplementedError

    def _convert_to_numpy_and_check(self, ids):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        if not isinstance(ids, list):
            raise TypeError(f"`ids` should be the list, but got type {type(ids)}.")
        return ids

    def decode(self,
               token_ids: Optional[Union[List[int], List[List[int]]]],
               skip_special_tokens: bool = False,
               **kwargs):
        """
        Convert the token ids to the string

        Args:
            token_ids(list[int], list[list[int]]):
            skip_special_tokens(bool): Whether to skip the special the token such as the CLS and EOS. Default False.
            **kwargs: Other kwargs passed to the internal function. Currently not used.

        Examples:
            >>> from mindformers import T5Tokenizer
            >>> tokenizer = T5Tokenizer.from_pretrained('t5_small')
            >>> ids = tokenizer.encode("hell world")
            >>> output = tokenizer.decode(ids)
            >>> print(output)
            hell world
            >>> ids = tokenizer.batch_encode_plus(["hello world", "nice to see you!"],
            ...                                   padding="max_length", max_length=8)["input_ids"]
            >>> output = tokenizer.decode(ids)
            >>> print(output)
            ['hello world', 'nice to see you!']


        Returns:
            A string or a list of strings.
        """
        if isinstance(token_ids, mindspore.Tensor):
            token_ids = token_ids.asnumpy().tolist()
        token_ids = self._convert_to_numpy_and_check(token_ids)
        if isinstance(token_ids[0], (list, np.ndarray)):
            output = []
            for line in token_ids:
                line = self._convert_to_numpy_and_check(line)
                new_strs = self._decode(line, skip_special_tokens=skip_special_tokens, **kwargs)
                output.append(new_strs)
        else:
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
        if isinstance(tokens, list):
            outstring = ""
            for token in tokens:
                outstring += " " + " ".join(token).strip()

            return outstring
        return " ".join(tokens).strip()

    def tokenize(self, text):
        raise NotImplementedError

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1):
        if token_ids_1:
            return token_ids_0 + token_ids_1
        return token_ids_0

    def num_special_tokens_to_add(self):
        """Return the special tokens to be added to the ids and ids_pair"""
        ids = []
        ids_pair = []
        output = self.build_inputs_with_special_tokens(ids, ids_pair)
        return len(output)

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
                           truncation=False,
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
        processed_output = self._batch_postprocess_ids(ids=text_ids,
                                                       pair_ids=text_pair_ids,
                                                       max_length=max_length,
                                                       truncation=truncation,
                                                       padding_strategy=padding_strategy,
                                                       add_special_tokens=add_special_tokens,
                                                       return_tensors=return_tensors,
                                                       return_token_type_ids=return_token_type_ids,
                                                       return_attention_mask=return_attention_mask,
                                                       return_batch=return_batch)
        return processed_output

    def _batch_postprocess_ids(self, ids,
                               pair_ids=None,
                               add_special_tokens=True,
                               max_length=None,
                               truncation=False,
                               padding_strategy="do_not_pad",
                               return_tensors=None,
                               return_token_type_ids=None,
                               return_attention_mask=None,
                               return_batch=True):
        """Convert the input_ids to the format of model inputs"""
        if return_tensors and return_tensors != 'ms':
            raise ValueError("You should set return_tensors to be `ms`.")
        if not return_batch and len(ids) != 1:
            raise ValueError(f"If `return_batch` is False, the length of input ids should be 1. But found {len(ids)}. "
                             f"Input ids is: {ids}. To fix this, you can set the return_batch=True")
        if pair_ids:
            paired_ids = zip(ids, pair_ids)
        else:
            paired_ids = zip(ids, [None] * len(ids))
        output = defaultdict(list)
        for per_ids, per_pair_ids in paired_ids:
            per_output = self.postprocess_ids(ids=per_ids,
                                              pair_ids=per_pair_ids,
                                              padding_strategy="do_not_pad",
                                              return_tensors=None,
                                              add_special_tokens=add_special_tokens,
                                              max_length=max_length,
                                              truncation=truncation,
                                              return_token_type_ids=return_token_type_ids,
                                              return_attention_mask=return_attention_mask)
            if not return_batch:
                output = per_output
            else:
                for k, v in per_output.items():
                    output[k].append(v)
        output_map = self._pad(output, max_length=max_length, padding_strategy=padding_strategy,
                               return_attention_mask=return_attention_mask)
        if return_tensors:
            for k in output_map.keys():
                v = np.array(output_map[k])
                if v.dtype == np.int64:
                    v = v.astype(np.int32)
                output_map[k] = Tensor(v)
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
