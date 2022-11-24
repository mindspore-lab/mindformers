import os

from mindspore.train.serialization import load_checkpoint, load_param_into_net

from .xformer_book import XFormerBook
from .models.base_config import BaseConfig
from .models.build_model import build_model_config, build_model
from .tools import logger
from .tools.register.config import XFormerConfig
from .tools.download_tools import downlond_with_progress_bar

class AutoConfig:
    _support_list = XFormerBook.get_model_support_list()

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        is_path = os.path.exists(pretrained_model_name_or_path)

        if is_path:
            if not pretrained_model_name_or_path.endswith(".yaml"):
                raise TypeError(f"{pretrained_model_name_or_path} should be a .yaml file for model config.")

            config_args = XFormerConfig(pretrained_model_name_or_path)
        else:
            model_type = pretrained_model_name_or_path.split('_')[0]
            if model_type not in cls._support_list.keys() or \
                    pretrained_model_name_or_path not in cls._support_list[model_type]:
                raise ValueError(f"{pretrained_model_name_or_path} is not a supported model type or a valied path to model config. "
                                 f"supported model could be selected from {cls._support_list}.")

            checkpoint_path = os.path.join(XFormerBook.get_default_checkpoint_download_folder(),
                                           model_type)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            yaml_file = os.path.join(checkpoint_path, pretrained_model_name_or_path + ".yaml")
            if not os.path.exists(yaml_file):
                url = XFormerBook.get_model_config_url_list()[pretrained_model_name_or_path][0]
                downlond_with_progress_bar(url, yaml_file)

            config_args = XFormerConfig(yaml_file)

        config = build_model_config(config_args.model.model_config)
        return config

    @classmethod
    def show_support_list(cls):
        logger.info(f"support list of {cls.__name__} is:")
        for key, val in cls._support_list.items():
            logger.info('   ', key, ':', val)
        logger.info("-------------------------------------")

