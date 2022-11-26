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

'''
BaseModel
'''
import os
import yaml

import mindspore as ms
from mindspore import nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from ..xformer_book import XFormerBook, print_path_or_list
from .build_model import build_model_config
from .base_config import BaseConfig
from ..tools.register import MindFormerConfig
from ..tools.download_tools import downlond_with_progress_bar
from ..tools import logger


class BaseModel(nn.Cell):
    '''
    BaseModel for all models.
    '''
    _support_list = []

    def _load_checkpoint(self, **kwargs):
        '''
        _load_checkpoint for load weights.
        (only support standalone mode, and distribute mode waits for developing)

        Args:
            kwargs:
                checkpoint_name_or_path (str): A string for supported model name or
                a path to model weights.
                The supported model name could be selected from .show_support_list method.
                set it to None, and does not load weights.

            other supposed parameters.
        '''
        checkpoint_name_or_path = kwargs.pop("checkpoint_name_or_path", None)

        if checkpoint_name_or_path is not None:
            is_path = os.path.exists(checkpoint_name_or_path)
            if is_path:
                param = load_checkpoint(checkpoint_name_or_path)
                logger.info("the given config and weights in %s are used for"
                            " model building.", checkpoint_name_or_path)
            else:
                if checkpoint_name_or_path not in self._support_list:
                    raise ValueError(f"{checkpoint_name_or_path} is not a supported default model,"
                                     f" please select from {self._support_list}.")

                ckpt_file = os.path.join(XFormerBook.get_default_checkpoint_download_folder(),
                                         checkpoint_name_or_path.split("_")[0],
                                         checkpoint_name_or_path+".ckpt")
                if not os.path.exists(ckpt_file):
                    url = XFormerBook.get_model_ckpt_url_list()[checkpoint_name_or_path]
                    downlond_with_progress_bar(url, ckpt_file)

                param = load_checkpoint(ckpt_file)
                logger.info("the given config and weights"
                            " in %s are used for model building.", ckpt_file)
            try:
                load_param_into_net(self, param)
            except RuntimeError:
                logger.error("the given config and weights in %s are"
                             " mismatched, and weights load failed", ckpt_file)
            logger.info("model built successfully!")

    def save_pretrained(self, save_directory=None, save_name="mindspore_model"):
        '''
        save_pretrained.
        (only support standalone mode, and distribute mode waits for developing)

        Args:
            save_directory (str): a directory to save model ckpt and config yaml

            save_name (str): the name of save files.
        '''
        if save_directory is None:
            save_directory = XFormerBook.get_default_checkpoint_save_folder()
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

        is_exit = os.path.exists(save_directory)
        if not is_exit:
            os.makedirs(save_directory)

        if not os.path.isdir(save_directory):
            raise ValueError(f"{save_directory} is not a directory.")

        checkpoint_path = os.path.join(save_directory, save_name + '.ckpt')
        config_path = os.path.join(save_directory, save_name + '.yaml')

        ms.save_checkpoint(self, checkpoint_path)
        if self.config is None:
            # A model should have "config" attribute for model save.
            raise AttributeError("the model has no config attribute.")

        parsed_config = self._inverse_parse_config(self.config)
        wraped_config = self._warap_config(parsed_config)

        with open(config_path, 'w') as file_pointer:
            file_pointer.write(yaml.dump(wraped_config))
        file_pointer.close()
        logger.info("model saved successfully!")

    def _inverse_parse_config(self, config):
        '''
        Inverse parse config method, which builds yaml file content for model config.

        Args:
            config (BaseConfig): a model config inherited from BaseConfig.

        Returns:
            A model config, which follows the yaml content.
        '''
        if not isinstance(config, BaseConfig):
            return config

        class_name = self.__class__.__name__
        config.update({"type": class_name})

        for key, val in config.items():
            new_val = self._inverse_parse_config(val)
            config.update({key: new_val})

        return config

    def _warap_config(self, config):
        '''
        Wrap config function, which wraps a config to rebuild content of yaml file.

        Args:
            config (BaseConfig): a config processed by _inverse_parse_config function.

        Returns:
            A (config) dict for yaml.dump.
        '''
        model_name = self.__class__.__name__
        return {"model": {"model_config": config.to_dict(), "arch": {"type": model_name}}}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_dir):
        '''
        From pretrain method, which instantiate a Model by pretrained model name or path.
        (only support standalone mode, and distribute mode waits for developing!)

        Args:
            pretrained_model_name_or_path (str): A supported model name or a
            directory to model checkpoint (including .yaml file for config
            and .ckpt file for weights), the supported model name could be
            selected from .show_support_list().

        Returns:
            A model, which inherited from BaseModel.
        '''
        if pretrained_model_name_or_dir is None:
            raise ValueError("a model cannot be built from pretrained"
                             " without pretrained_model_name_or_dir.")

        is_exist = os.path.exists(pretrained_model_name_or_dir)
        is_dir = os.path.isdir(pretrained_model_name_or_dir)

        if not is_exist and (pretrained_model_name_or_dir not in cls._support_list):
            raise ValueError(f'{pretrained_model_name_or_dir} does not exist,'
                             f' or it is not supported by {cls.__name__}. '
                             f'please select from {cls._support_list}.')
        if is_exist and not is_dir:
            raise ValueError(f"{pretrained_model_name_or_dir} is not a directory.")

        if is_dir:
            yaml_list = [file for file in os.listdir(pretrained_model_name_or_dir)
                         if file.endswith(".yaml")]
            ckpt_list = [file for file in os.listdir(pretrained_model_name_or_dir)
                         if file.endswith(".ckpt")]
            if not yaml_list or not ckpt_list:
                raise FileNotFoundError(f"there is no yaml file for model config or ckpt file "
                                        f"for model weights in {pretrained_model_name_or_dir}.")
            yaml_file = os.path.join(pretrained_model_name_or_dir, yaml_list[0])
            ckpt_file = os.path.join(pretrained_model_name_or_dir, ckpt_list[0])
            logger.info("config in %s and weights in %s are used for "
                        "model building.", yaml_file, ckpt_file)

            config_args = MindFormerConfig(yaml_file)
            config = build_model_config(config_args.model.model_config)
            model = cls(config)
            param = load_checkpoint(ckpt_file)
            try:
                load_param_into_net(model, param)
            except RuntimeError:
                logger.error("config in %s and weights in %s are"
                             " mismatched, and weights load failed", yaml_file, ckpt_file)
            logger.info("model built successfully!")
        else:
            checkpoint_path = os.path.join(XFormerBook.get_default_checkpoint_download_folder(),
                                           pretrained_model_name_or_dir.split("_")[0])
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            yaml_file = os.path.join(checkpoint_path, pretrained_model_name_or_dir+".yaml")
            ckpt_file = os.path.join(checkpoint_path, pretrained_model_name_or_dir+".ckpt")

            if not os.path.exists(ckpt_file):
                url = XFormerBook.get_model_ckpt_url_list()[pretrained_model_name_or_dir][0]
                downlond_with_progress_bar(url, ckpt_file)

            if not os.path.exists(yaml_file):
                url = XFormerBook.get_model_config_url_list()[pretrained_model_name_or_dir][0]
                downlond_with_progress_bar(url, yaml_file)

            logger.info("config in %s and weights in %s are"
                        " used for model building.", yaml_file, ckpt_file)
            config_args = MindFormerConfig(yaml_file)
            config = build_model_config(config_args.model.model_config)
            model = cls(config)

            param = load_checkpoint(ckpt_file)
            load_param_into_net(model, param)
            logger.info("model built successfully!")
        return model

    @classmethod
    def show_support_list(cls):
        '''show_support_list'''
        logger.info("support list of %s is:", cls.__name__)
        print_path_or_list(cls._support_list)
