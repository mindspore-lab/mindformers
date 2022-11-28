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
"""Build Model API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_model(
        config: dict = None, default_args: dict = None,
        module_type: str = 'models', class_name: str = None, **kwargs):
    """Build model API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        if default_args is None:
            default_args = {}

        model_config = build_model_config(config.model_config, default_args=default_args)

        if model_config is not None:
            if default_args is not None:
                for key, value in default_args.items():
                    model_config.__setattr__(key, value)
                default_args = {}
            default_args.setdefault('config', model_config)

            return MindFormerRegister.get_instance_from_cfg(
                config.arch, MindFormerModuleType.MODELS, default_args=default_args)
        return None
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def build_tokenizer(
        config: dict = None, default_args: dict = None,
        module_type: str = 'tokenizer', class_name: str = None, **kwargs):
    """Build tokenizer API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.TOKENIZER, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def build_encoder(
        config: dict = None, default_args: dict = None,
        module_type: str = 'encoder', class_name: str = None, **kwargs):
    """Build encoder API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        cfg_encoders = config
        if not isinstance(cfg_encoders, list):
            return MindFormerRegister.get_instance_from_cfg(
                cfg_encoders, MindFormerModuleType.ENCODER, default_args=default_args)
        encoders = []
        for encoder in cfg_encoders:
            encoder_op = MindFormerRegister.get_instance_from_cfg(
                encoder, MindFormerModuleType.ENCODER)
            encoders.append(encoder_op)
        return encoders
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def build_head(
        config: dict = None, default_args: dict = None,
        module_type: str = 'head', class_name: str = None, **kwargs):
    """Build head API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        cfg_heads = config
        if not isinstance(cfg_heads, list):
            return MindFormerRegister.get_instance_from_cfg(
                cfg_heads, MindFormerModuleType.HEAD, default_args=default_args)
        heads = []
        for head in cfg_heads:
            head_op = MindFormerRegister.get_instance_from_cfg(
                head, MindFormerModuleType.HEAD)
            heads.append(head_op)
        return heads
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def build_model_config(
        config: dict = None, default_args: dict = None,
        module_type: str = 'config', class_name: str = None, **kwargs):
    """Build model config API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        if config.text_config is not None:
            config.text_config = build_model_config(config.text_config)
        if config.vision_config is not None:
            config.vision_config = build_model_config(config.vision_config)
        if config.head_config is not None:
            config.head_config = build_model_config(config.head_config)
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.CONFIG, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, *args, **kwargs)
