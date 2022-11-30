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
AutoConfig、AutoModel
'''
import os

from mindspore.train.serialization import load_checkpoint, load_param_into_net

from .processor import build_feature_extractor
from .mindformer_book import MindFormerBook, print_dict
from .models.base_config import BaseConfig
from .models.build_model import build_model
from .models.build_config import build_model_config
from .tools import logger
from .tools.register.config import MindFormerConfig
from .tools.download_tools import downlond_with_progress_bar


class AutoConfig:
    ''' AutoConfig '''
    _support_list = MindFormerBook.get_model_support_list()

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        '''
        From pretrain method, which instantiate a Config by pretrained model name or path.

        Args:
            pretrained_model_name_or_path (str): A supported model name or a path to model
            config (.yaml), the supported model name could be selected from
            AutoConfig.show_support_list().

        Returns:
            A model config, which inherited from BaseConfig.
        '''
        is_path = os.path.exists(pretrained_model_name_or_path)

        if is_path:
            if not pretrained_model_name_or_path.endswith(".yaml"):
                raise TypeError(f"{pretrained_model_name_or_path} should be a .yaml file for model"
                                " config.")

            config_args = MindFormerConfig(pretrained_model_name_or_path)
        else:
            model_type = pretrained_model_name_or_path.split('_')[0]
            if model_type not in cls._support_list.keys() or \
                    pretrained_model_name_or_path not in cls._support_list[model_type]:
                raise ValueError(f"{pretrained_model_name_or_path} is not a supported"
                                 f" model type or a valid path to model config."
                                 f" supported model could be selected from {cls._support_list}.")

            checkpoint_path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                           model_type)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            yaml_file = os.path.join(checkpoint_path, pretrained_model_name_or_path + ".yaml")
            if not os.path.exists(yaml_file):
                url = MindFormerBook.get_model_config_url_list()[pretrained_model_name_or_path][0]
                downlond_with_progress_bar(url, yaml_file)

            config_args = MindFormerConfig(yaml_file)

        config = build_model_config(config_args.model.model_config)
        return config

    @classmethod
    def show_support_list(cls):
        '''show support list method'''
        logger.info("support list of %s is:", cls.__name__)
        print_dict(cls._support_list)


class AutoModel:
    ''' AutoModel '''
    _support_list = MindFormerBook.get_model_support_list()

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_dir)` method "
            "or `AutoModel.from_config(config)` method."
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        '''
        From config method, which instantiate a Model by config.

        Args:
            config (str, BaseConfig): A string for supported model name, or a model config
            inherited from BaseConfig, the supported model name could be selected from
            AutoModel.show_support_list().

            kwargs:
                  checkpoint_name_or_path (str): A string for supported model name, or a path to
                  ckpt file, which allows to load pretrained weights. The supported model name
                  could be selected from AutoModel.show_support_list().

                  other supposed parameters.
        Returns:
            A model, which inherited from BaseModel.
        '''
        if config is None:
            raise ValueError("a model cannot be built from config with config is None.")

        if isinstance(config, BaseConfig):
            inversed_config = cls._inverse_parse_config(config)
            config_args = cls._wrap_config(inversed_config)
        elif os.path.exists(config) and config.endswith(".yaml"):
            config_args = MindFormerConfig(config)
        else:
            raise TypeError("config should be inherited from BaseConfig,"
                            " or a path to .yaml file for model config.")

        model = build_model(config_args.model)

        checkpoint_name_or_path = kwargs.pop("checkpoint_name_or_path", None)
        if checkpoint_name_or_path is not None:
            is_path = os.path.exists(checkpoint_name_or_path)
            if is_path:
                param = load_checkpoint(checkpoint_name_or_path)
                logger.info("the given config and weights in %s are used for model"
                            " building.", checkpoint_name_or_path)
            else:
                model_type = checkpoint_name_or_path.split('_')[0]
                if model_type not in cls._support_list.keys() or \
                        checkpoint_name_or_path not in cls._support_list[model_type]:
                    raise ValueError(f"{checkpoint_name_or_path} is not a supported model"
                                     f" type or a valid path to model weights. supported "
                                     f"model could be selected from {cls._support_list}.")

                ckpt_file = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                         model_type, checkpoint_name_or_path+".ckpt")
                if not os.path.exists(ckpt_file):
                    url = MindFormerBook.get_model_ckpt_url_list()[checkpoint_name_or_path]
                    downlond_with_progress_bar(url, ckpt_file)

                param = load_checkpoint(ckpt_file)
                logger.info("the given config and weights in %s are used for"
                            " model building.", ckpt_file)
            try:
                load_param_into_net(model, param)
            except RuntimeError:
                logger.error("the weights in %s are mismatched with the model"
                             " config, and weights load failed", checkpoint_name_or_path)
            logger.info("model built successfully!")
        return model

    @classmethod
    def _inverse_parse_config(cls, config):
        '''
        Inverse parse config method, which builds yaml file content for model config.

        Args:
            config (BaseConfig): a model config inherited from BaseConfig.

        Returns:
            A model config, which follows the yaml content.
        '''
        if not isinstance(config, BaseConfig):
            return config

        class_name = config.__class__.__name__
        config.update({"type": class_name})

        for key, val in config.items():
            new_val = cls._inverse_parse_config(val)
            config.update({key: new_val})

        return config

    @classmethod
    def _wrap_config(cls, config):
        '''
        Wrap config function, which wraps a config to rebuild content of yaml file.

        Args:
            config (BaseConfig): a config processed by _inverse_parse_config function.

        Returns:
            A model config, which has the same content as a yaml file.
        '''
        model_name = config.type.split("Config")[0] + "Model"
        arch = BaseConfig(type=model_name)
        model = BaseConfig(model_config=config, arch=arch)
        return BaseConfig(model=model)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_dir):
        '''
        From pretrain method, which instantiate a Model by pretrained model name or path.

        Args:
            pretrained_model_name_or_path (str): A supported model name or a
            directory to model checkpoint (including .yaml file for config
            and .ckpt file for weights), the supported model name could be
            selected from AutoModel.show_support_list().

        Returns:
            A model, which inherited from BaseModel.
        '''
        if pretrained_model_name_or_dir is None:
            raise ValueError("a model cannot be built from pretrained without"
                             " pretrained_model_name_or_dir.")

        is_exist = os.path.exists(pretrained_model_name_or_dir)
        is_dir = os.path.isdir(pretrained_model_name_or_dir)

        if is_exist:
            if not is_dir:
                raise ValueError(f"{pretrained_model_name_or_dir} is not a directory.")
        else:
            model_type = pretrained_model_name_or_dir.split('_')[0]
            if model_type not in cls._support_list.keys() or pretrained_model_name_or_dir \
                    not in cls._support_list[model_type]:
                raise ValueError(f"{pretrained_model_name_or_dir} is not a supported model"
                                 f" type or a valid path to model config. supported model"
                                 f" could be selected from {cls._support_list}.")

        if is_dir:
            yaml_list = [file for file in os.listdir(pretrained_model_name_or_dir)
                         if file.endswith(".yaml")]
            ckpt_list = [file for file in os.listdir(pretrained_model_name_or_dir)
                         if file.endswith(".ckpt")]
            if not yaml_list or not ckpt_list:
                raise FileNotFoundError(f"there is no yaml file for model config or ckpt file"
                                        f" for model weights in {pretrained_model_name_or_dir}")

            yaml_file = os.path.join(pretrained_model_name_or_dir, yaml_list[0])
            ckpt_file = os.path.join(pretrained_model_name_or_dir, ckpt_list[0])
            logger.info("config in %s and weights in %s are used for model"
                        " building.", yaml_file, ckpt_file)

            config_args = MindFormerConfig(yaml_file)
            model = build_model(config_args.model)
            param = load_checkpoint(ckpt_file)
            try:
                load_param_into_net(model, param)
            except RuntimeError:
                logger.error("config in %s and weights in %s are"
                             " mismatched, and weights load failed", yaml_file, ckpt_file)
            logger.info("model built successfully!")
        else:
            checkpoint_path = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                           pretrained_model_name_or_dir.split("_")[0])
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            yaml_file = os.path.join(checkpoint_path, pretrained_model_name_or_dir+".yaml")
            ckpt_file = os.path.join(checkpoint_path, pretrained_model_name_or_dir+".ckpt")

            if not os.path.exists(ckpt_file):
                url = MindFormerBook.get_model_ckpt_url_list()[pretrained_model_name_or_dir][0]
                downlond_with_progress_bar(url, ckpt_file)

            if not os.path.exists(yaml_file):
                url = MindFormerBook.get_model_config_url_list()[pretrained_model_name_or_dir][0]
                downlond_with_progress_bar(url, yaml_file)

            logger.info("config in %s and weights in %s are used for model"
                        " building.", yaml_file, ckpt_file)
            config_args = MindFormerConfig(yaml_file)
            model = build_model(config_args.model)
            param = load_checkpoint(ckpt_file)
            load_param_into_net(model, param)
            logger.info("model built successfully!")
        return model

    @classmethod
    def show_support_list(cls):
        '''show support list method'''
        logger.info("support list of %s is:", cls.__name__)
        print_dict(cls._support_list)


class AutoFeatureExtractor:
    ''' AutoFeatureExtractor '''
    _support_list = MindFormerBook.get_model_support_list()

    def __init__(self):
        raise EnvironmentError(
            "AutoFeatureExtractor is designed to be instantiated "
            "using the `AutoFeatureExtractor.from_pretrained(yaml_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, yaml_name_or_path):
        '''
        From pretrain method, which instantiate a feature extractor by yaml name or path.

        Args:
            yaml_name_or_path (str): A supported yaml name or a path to .yaml file,
            the supported model name could be selected from .show_support_list().

        Returns:
            A feature extractor which inherited from BaseFeatureExtractor.
        '''
        if yaml_name_or_path is None:
            raise ValueError("a feature extractor cannot be built from pretrained"
                             " without yaml_name_or_path.")

        is_exist = os.path.exists(yaml_name_or_path)
        model_name = yaml_name_or_path.split("_")[0]
        if not is_exist and model_name not in cls._support_list.keys():
            raise ValueError(f'{yaml_name_or_path} does not exist,'
                             f' and it is not supported by {cls.__name__}. '
                             f'please select from {cls._support_list}.')

        if is_exist:
            logger.info("config in %s is used for feature extractor"
                        " building.", yaml_name_or_path)

            config_args = MindFormerConfig(yaml_name_or_path)
        else:
            if model_name in cls._support_list.keys() and\
                    yaml_name_or_path in cls._support_list[model_name]:
                checkpoint_path = os.path.join(
                    MindFormerBook.get_default_checkpoint_download_folder(), model_name)
            else:
                raise ValueError(f'{yaml_name_or_path} does not exist,'
                                 f' or it is not supported by {cls.__name__}.'
                                 f' please select from {cls._support_list}.')

            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            yaml_file = os.path.join(checkpoint_path, yaml_name_or_path + ".yaml")
            if not os.path.exists(yaml_file):
                url = MindFormerBook.get_model_config_url_list()[yaml_name_or_path][0]
                downlond_with_progress_bar(url, yaml_file)
            logger.info("config in %s is used for feature extractor"
                        " building.", yaml_file)

            config_args = MindFormerConfig(yaml_file)

        feature_extractor = build_feature_extractor(config_args.processor.feature_extractor)
        logger.info("feature extractor built successfully!")
        return feature_extractor

    @classmethod
    def show_support_list(cls):
        '''show support list method'''
        logger.info("support list of %s is:", cls.__name__)
        print_dict(cls._support_list)
