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

"""Build Tokenizer API."""
import os.path

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig
from ..mindformer_book import MindFormerBook


def check_and_add_vocab_file_path(config, **kwargs):
    """And the vocab file path to the config if there is not vocab file in the config"""
    if 'vocab_file' in config:
        return
    class_name = config['type']
    dynamic_class = MindFormerRegister.get_cls(module_type='tokenizer', class_name=class_name)
    # If the tokenizer does not require the vocab_file, just stop
    if not dynamic_class.vocab_files_names:
        return
    name_or_path = class_name
    path = kwargs.pop('lib_path', None)
    remote_tokenizer_support_list = MindFormerBook.get_tokenizer_url_support_list().keys()
    if name_or_path not in remote_tokenizer_support_list and path:
        read_vocab_file_dict, read_tokenizer_file_dict = \
            dynamic_class.read_files_according_specific_by_tokenizer(name_or_path=path)
        config.update(read_vocab_file_dict)
        config.update(read_tokenizer_file_dict)
    else:
        if not hasattr(config, "vocab_file") or config.vocab_file is None:
            raise ValueError("tokenizer.vocab_file in yaml file is not set, "
                             "please set tokenizer.vocab_file a correct value.")
        if not os.path.exists(config.vocab_file):
            raise ValueError(f"{config.vocab_file} is not existed, "
                             f"please check vocab_file in yaml and set a correct value.")


def build_tokenizer(
        config: dict = None, default_args: dict = None, module_type: str = 'tokenizer',
        class_name: str = None, use_legacy: bool = True, **kwargs):
    r"""Build tokenizer For MindFormer.
    Instantiate the tokenizer from MindFormerRegister's registry.

    Args:
        config (dict): The task tokenizer's config. Default: None.
        default_args (dict): The default argument of tokenizer API. Default: None.
        module_type (str): The module type of MindFormerModuleType. Default: 'tokenizer'.
        class_name (str): The class name of tokenizer API. Default: None.
        use_legacy (bool): Distinguish whether to use hf-tokenizer or not. Default: True.

    Return:
        The function instance of tokenizer API.

    Examples:
        >>> from mindformers import build_tokenizer
        >>> from mindformers.tools.register import MindFormerConfig
        >>> config = MindFormerConfig('configs/clip/run_clip_vit_b_32_pretrain_flickr8k.yaml')
        >>> tokenizer_from_config = build_tokenizer(config.processor.tokenizer)
    """
    if not use_legacy:
        from transformers import AutoTokenizer
        pretrained_model_dir = kwargs.get("pretrained_model_dir", None)
        if not pretrained_model_dir:
            raise ValueError("The current interface supports passing a local folder path, "
                             "but the provided path is empty or None.")
        pretrained_model_dir = os.path.realpath(pretrained_model_dir)
        if not os.path.isdir(pretrained_model_dir):
            raise ValueError(f"The current interface supports passing a local folder path, "
                             f"but the provided path '{pretrained_model_dir}' is not a valid directory.")

        trust_remote_code = kwargs.get("trust_remote_code", False)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_dir,
                                                  trust_remote_code=trust_remote_code)
        return tokenizer

    if config is None and class_name is None:
        return None
    if config is not None:
        if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
            config = MindFormerConfig(**config)
        if 'auto_register' in config:
            MindFormerRegister.auto_register(class_reference=config.pop('auto_register'), module_type=module_type)
        check_and_add_vocab_file_path(config, **kwargs)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.TOKENIZER, default_args=default_args)

    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
