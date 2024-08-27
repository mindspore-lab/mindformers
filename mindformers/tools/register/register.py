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
# This file was refer to project
# https://gitee.com/mindspore/vision/blob/master/mindvision/engine/class_factory.py
# ============================================================================
""" Class Register Module For MindFormers."""

import inspect
import os

from mindformers.tools.hub.dynamic_module_utils import get_class_from_dynamic_module


class MindFormerModuleType:
    """Class module type for vision pretrain"""

    def __init__(self):
        pass

    TRAINER = 'trainer'
    PIPELINE = 'pipeline'
    PROCESSOR = 'processor'
    TOKENIZER = 'tokenizer'
    DATASET = 'dataset'
    MASK_POLICY = 'mask_policy'
    DATASET_LOADER = 'dataset_loader'
    DATASET_SAMPLER = 'dataset_sampler'
    TRANSFORMS = 'transforms'
    ENCODER = 'encoder'
    MODELS = 'models'
    MODULES = 'modules'
    TRANSFORMER = 'transformer'
    BASE_LAYER = 'base_layer'
    CORE = 'core'
    HEAD = 'head'
    LOSS = 'loss'
    LR = 'lr'
    OPTIMIZER = 'optimizer'
    CONTEXT = 'context'
    CALLBACK = 'callback'
    WRAPPER = 'wrapper'
    METRIC = 'metric'
    CONFIG = 'config'
    TOOLS = 'tools'
    FEATURE_EXTRACTOR = 'feature_extractor'
    DATA_HANDLER = 'data_handler'


class MindFormerRegister:
    """
    Module class factory for ring-mo.
    """

    def __init__(self):
        pass

    registry = {}

    @classmethod
    def register(cls, module_type=MindFormerModuleType.TOOLS, alias=None):
        """Register class into registry
        Args:
            module_type (ModuleType) :
                module type name, default ModuleType.GENERAL
            alias (str) : class alias

        Returns:
            wrapper
        """

        def wrapper(register_class):
            """Register-Class with wrapper function.

            Args:
                register_class : class need to register

            Returns:
                wrapper of register_class
            """
            class_name = alias if alias is not None else register_class.__name__
            if module_type not in cls.registry:
                cls.registry[module_type] = {class_name: register_class}
            else:
                cls.registry[module_type][class_name] = register_class
            return register_class

        return wrapper

    @classmethod
    def register_cls(cls, register_class, module_type=MindFormerModuleType.TOOLS, alias=None):
        """Register class with type name.

        Args:
            register_class : class need to register
            module_type :  module type name, default ModuleType.COMMON
            alias : class name
        """
        class_name = alias if alias is not None else register_class.__name__
        if module_type not in cls.registry:
            cls.registry[module_type] = {class_name: register_class}
        else:
            cls.registry[module_type][class_name] = register_class
        return register_class

    @classmethod
    def is_exist(cls, module_type, class_name=None):
        """Determine whether class name is in the current type group.

        Args:
            module_type : Module type
            class_name : class name

        Returns:
            True/False
        """
        if not class_name:
            return module_type in cls.registry
        registered = module_type in cls.registry and class_name in cls.registry.get(module_type)
        return registered

    @classmethod
    def get_cls(cls, module_type, class_name=None):
        """Get class

        Args:
            module_type : Module type
            class_name : class name

        Returns:
            register_class
        """
        if not cls.is_exist(module_type, class_name):
            raise ValueError("Can't find class type {} class name {} \
            in class registry".format(module_type, class_name))

        if not class_name:
            raise ValueError(
                "Can't find class. class type = {}".format(class_name))
        register_class = cls.registry.get(module_type).get(class_name)
        return register_class

    @classmethod
    def get_instance_from_cfg(cls, cfg, module_type=MindFormerModuleType.TOOLS, default_args=None):
        """Get instance.
        Args:
            cfg (dict) : Config dict. It should at least contain the key "type".
            module_type : module type
            default_args (dict, optional) : Default initialization arguments.

        Returns:
            obj : The constructed object.
        """
        if not isinstance(cfg, dict):
            raise TypeError(
                "Cfg must be a Config, but got {}".format(type(cfg))
            )

        if 'auto_register' in cfg:
            cls.auto_register(class_reference=cfg.pop('auto_register'), module_type=module_type)

        if 'type' not in cfg:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type",'
                'but got {}\n{}'.format(cfg, default_args)
            )
        if not (isinstance(default_args, dict) or default_args is None):
            raise TypeError(
                'default_args must be a dict or None'
                'but got {}'.format(type(default_args)))

        args = cfg.copy()
        if default_args is not None:
            for k, v in default_args.items():
                if k not in args:
                    args.setdefault(k, v)
                else:
                    args[k] = v

        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = cls.get_cls(module_type, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise ValueError("Can't find class type {} class name {} \
            in class registry".format(type, obj_type))

        try:
            return obj_cls(**args)
        except Exception as e:
            raise type(e)('{}: {}'.format(obj_cls.__name__, e))

    @classmethod
    def get_instance(cls, module_type=MindFormerModuleType.TOOLS, class_name=None, **kwargs):
        """Get instance.
        Args:
            module_type : module type
            class_name : class type
        Returns:
            object : The constructed object
        """
        if class_name is None:
            raise ValueError("Class name cannot be None.")

        if isinstance(class_name, str):
            obj_cls = cls.get_cls(module_type, class_name)
        elif inspect.isclass(class_name):
            obj_cls = class_name
        else:
            raise ValueError("Can't find class type {} class name {} \
            in class registry.".format(type, class_name))

        try:
            return obj_cls(**kwargs)
        except Exception as e:
            raise type(e)('{}: {}'.format(obj_cls.__name__, e))

    @classmethod
    def auto_register(cls, class_reference: str, module_type=MindFormerModuleType.TOOLS):
        """Auto register function.
        Args:
            class_reference: The full name of the class to load.
            module_type : module type.
        """
        if not isinstance(class_reference, str):
            raise ValueError(f"auto_map must be the type of string, but get {type(class_reference)} ."
                             f"Please fill in the following format: module_file.function_name, such as,"
                             f"llama_model.LlamaForCausalLM")
        register_path = os.getenv("REGISTER_PATH", '')
        if not register_path:
            raise EnvironmentError("When configuring the 'auto_map' automatic registration function, "
                                   "REGISTER_PATH must be specified. "
                                   "It is recommended to complete this action"
                                   "through the official startup script "
                                   "'run_mindformer.py --register_path=module_file_path' "
                                   "or use 'export REGISTER_PATH=module_file_path' to complete this action.")
        if not os.path.realpath(register_path):
            raise EnvironmentError(f"REGISTER_PATH must be real path, but get {register_path}, "
                                   f"please specify the correct directory path.")
        register_path = os.path.realpath(os.getenv("REGISTER_PATH"))
        module_class = get_class_from_dynamic_module(
            class_reference=class_reference, pretrained_model_name_or_path=register_path)
        cls.register_cls(module_class, module_type=module_type)
