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
""" Class Register Module For MindFormers."""

import inspect
import os

from mindformers.core.context import is_legacy_model
from mindformers.tools.hub.dynamic_module_utils import get_class_from_dynamic_module


NEW_CLASS_PREFIX = "mcore_"


class MindFormerModuleType:
    """
    Enumerated class of the MindFormers module type, which includes:

    .. list-table::
        :widths: 25 25
        :header-rows: 1

        * - Enumeration Name
          - Value
        * - CALLBACK
          - 'CALLBACK'
        * - CONFIG
          - 'config'
        * - CONTEXT
          - 'context'
        * - DATASET
          - 'dataset'
        * - DATASET_LOADER
          - 'dataset_loader'
        * - DATASET_SAMPLER
          - 'dataset_sampler'
        * - DATA_HANDLER
          - 'data_handler'
        * - ENCODER
          - 'encoder'
        * - FEATURE_EXTRACTOR
          - 'feature_extractor'
        * - LOSS
          - 'loss'
        * - LR
          - 'lr'
        * - MASK_POLICY
          - 'mask_policy'
        * - METRIC
          - 'metric'
        * - MODELS
          - 'models'
        * - OPTIMIZER
          - 'optimizer'
        * - PIPELINE
          - 'pipeline'
        * - PROCESSOR
          - 'processor'
        * - TOKENIZER
          - 'tokenizer'
        * - TOOLS
          - 'tools'
        * - TRAINER
          - 'trainer'
        * - TRANSFORMS
          - 'transforms'
        * - WRAPPER
          - 'wrapper'

    Examples:
        >>> from mindformers.tools import MindFormerModuleType
        >>>
        >>> print(MindFormerModuleType.MODULES)
        modules
    """

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
    The registration interface for MindFormers, provides methods for registering and obtaining the interface.

    Examples:
        >>> from mindformers.tools import MindFormerModuleType, MindFormerRegister
        >>>
        >>>
        >>> # Using decorator to register the class
        >>> @MindFormerRegister.register(MindFormerModuleType.CONFIG)
        >>> class MyConfig:
        ...     def __init__(self, param):
        ...         self.param = param
        >>>
        >>>
        >>> # Using method to register the class
        >>> MindFormerRegister.register_cls(register_class=MyConfig, module_type=MindFormerRegister)
        >>>
        >>> print(MindFormerRegister.is_exist(module_type=MindFormerModuleType.CONFIG, class_name="MyConfig"))
        True
        >>> cls = MindFormerRegister.get_cls(module_type=MindFormerModuleType.CONFIG, class_name="MyConfig")
        >>> print(cls.__name__)
        MyConfig
        >>> instance = MindFormerRegister.get_instance_from_cfg(cfg={'type': 'MyConfig', 'param': 0},
        ...                                                     module_type=MindFormerModuleType.CONFIG)
        >>> print(instance.__class__.__name__)
        MyConfig
        >>> print(instance.param)
        0
        >>> instance = MindFormerRegister.get_instance(module_type=MindFormerModuleType.CONFIG,
        ...                                            class_name="MyConfig",
        ...                                            param=0)
        >>> print(instance.__class__.__name__)
        MyConfig
        >>> print(instance.param)
        0
    """

    def __init__(self):
        pass

    registry = {}
    search_names_map = {}

    @classmethod
    def register(cls, module_type=MindFormerModuleType.TOOLS, alias=None, legacy=True, search_names=None):
        """
        A decorator that registers the class in the registry.

        Args:
            module_type (MindFormerModuleType, optional): Module type name of MindFormers.
                Default: ``MindFormerModuleType.TOOLS``.
            alias (str, optional): Alias for the class. Default: ``None``.
            legacy (bool, optional): Legacy Class or not. Default: ``True``.
            search_names (Union[str, tuple, list, set], optional): mapping search_names to a class_name.
                Default: ``None``.
        Returns:
            Wrapper, decorates the registered class.
        """

        def wrapper(register_class):
            """Register-Class with wrapper function.

            Args:
                register_class : class need to register

            Returns:
                wrapper of register_class
            """
            class_name = alias if alias is not None else register_class.__name__
            class_name = cls._add_class_name_prefix(module_type, class_name, legacy)
            if module_type not in cls.registry:
                cls.registry[module_type] = {class_name: register_class}
            else:
                cls.registry[module_type][class_name] = register_class
            names = set()
            if search_names is not None:
                if isinstance(search_names, str):
                    names.add(search_names)
                elif isinstance(search_names, (list, tuple, set)):
                    names.update(search_names)
            names.add(class_name)
            for search_name in names:
                search_name = cls._add_class_name_prefix(module_type, search_name, legacy)
                cls.search_names_map[(module_type, search_name)] = class_name
            return register_class

        return wrapper

    @classmethod
    def register_cls(
            cls, register_class, module_type=MindFormerModuleType.TOOLS,
            alias=None, legacy=True, search_names=None
    ):
        """
        A method that registers a class into the registry.

        Args:
            register_class (type): The class that need to be registered.
            module_type (MindFormerModuleType, optional): Module type name of MindFormers.
                Default: ``MindFormerModuleType.TOOLS``.
            alias (str, optional): Alias for the class. Default: ``None``.
            legacy (bool, optional): Legacy Class or not. Default: ``True``.
            search_names (Union[str, tuple, list, set], optional): mapping search_names to a class_name.
                Default: ``None``.

        Returns:
            Class, the registered class itself.
        """
        class_name = alias if alias is not None else register_class.__name__
        class_name = cls._add_class_name_prefix(module_type, class_name, legacy)
        if module_type not in cls.registry:
            cls.registry[module_type] = {class_name: register_class}
        else:
            cls.registry[module_type][class_name] = register_class
        names = set()
        if search_names is not None:
            if isinstance(search_names, str):
                names.add(search_names)
            elif isinstance(search_names, (list, tuple, set)):
                names.update(search_names)
        names.add(class_name)
        for search_name in names:
            search_name = cls._add_class_name_prefix(module_type, search_name, legacy)
            cls.search_names_map[(module_type, search_name)] = class_name
        return register_class

    @classmethod
    def is_exist(cls, module_type, class_name=None):
        """
        Determines whether the given class name is in the current type group. If `class_name` is not given,
        determines if the given class name is in the current registered dictionary.

        Args:
            module_type (MindFormerModuleType): Module type name of MindFormers.
            class_name (str, optional): Class name. Default: ``None``.

        Returns:
            A boolean value, indicating whether it exists or not.
        """
        if not class_name:
            return module_type in cls.registry
        class_name = cls._add_class_name_prefix(module_type, class_name, is_legacy_model())
        if (module_type, class_name) in cls.search_names_map:
            class_name = cls.search_names_map[(module_type, class_name)]
            return module_type in cls.registry and class_name in cls.registry.get(module_type)
        registered = module_type in cls.registry and class_name in cls.registry.get(module_type)
        return registered

    @classmethod
    def get_cls(cls, module_type, class_name=None):
        """
        Get the class from the registry.

        Args:
            module_type (MindFormerModuleType): Module type name of MindFormers.
            class_name (str, optional): Class name. Default: ``None``.

        Returns:
            A registered class.

        Raises:
            ValueError: Can't find class `class_name` of type `module_type` in the registry.
            ValueError: Can't find type `module_type` in the registry.
        """
        if not cls.is_exist(module_type, class_name):
            raise ValueError(f"Can't find class type {module_type} class name {class_name} in class registry "
                             f"when use_legacy={is_legacy_model()}")

        if not class_name:
            raise ValueError(f"Can't find class. class type = {class_name}")
        class_name = cls._add_class_name_prefix(module_type, class_name, is_legacy_model())
        if (module_type, class_name) in cls.search_names_map:
            class_name = cls.search_names_map[(module_type, class_name)]
        if not (module_type in cls.registry and class_name in cls.registry.get(module_type)):
            raise ValueError(f"Can't find class type {module_type} class name {class_name} in class registry "
                             f"when use_legacy={is_legacy_model()}")
        register_class = cls.registry.get(module_type).get(class_name)
        return register_class

    @classmethod
    def get_instance_type_from_cfg(cls, cfg, module_type=MindFormerModuleType.MODELS):
        """
        Get instance's type of the class in the registry via configuration.

        Args:
            cfg (dict): Configuration dictionary. It should contain at least the key "type".
            module_type (MindFormerModuleType, optional): Module type name of MindFormers.
                Default: ``MindFormerModuleType.TOOLS``.
            default_args (dict, optional): Default initialization arguments. Default: ``None``.

        Returns:
            An instance of the class.

        Raises:
            TypeError: `cfg` must be a configuration.
            KeyError: `cfg` or `default_args` must contain the key "type".
            TypeError: `default_args` must be a dictionary or ``None``.
            ValueError: Can't find class `class_name` of type `module_type` in the registry.
        """
        if module_type == MindFormerModuleType.CONFIG:
            model_type = cfg.get('model_type', None)
            obj_type = cls.get_cls(module_type, model_type)
        elif module_type == MindFormerModuleType.MODELS:
            architectures = cfg.pop('architectures')
            if isinstance(architectures, list):
                obj_type = architectures[0]
            elif isinstance(architectures, str):
                obj_type = architectures
            else:
                raise ValueError("The type of model_config.architectures should be str or list of str.")
        else:
            obj_type = cfg.pop('type')
        return obj_type

    @classmethod
    def get_instance_from_cfg(cls, cfg, module_type=MindFormerModuleType.TOOLS, default_args=None):
        """
        Get instances of the class in the registry via configuration.

        Args:
            cfg (dict): Configuration dictionary. It should contain at least the key "type".
            module_type (MindFormerModuleType, optional): Module type name of MindFormers.
                Default: ``MindFormerModuleType.TOOLS``.
            default_args (dict, optional): Default initialization arguments. Default: ``None``.

        Returns:
            An instance of the class.

        Raises:
            TypeError: `cfg` must be a configuration.
            KeyError: `cfg` or `default_args` must contain the key "type".
            TypeError: `default_args` must be a dictionary or ``None``.
            ValueError: Can't find class `class_name` of type `module_type` in the registry.
        """
        if not isinstance(cfg, dict):
            raise TypeError(
                f"Cfg must be a Config, but got {type(cfg)}"
            )

        if 'auto_register' in cfg:
            cls.auto_register(class_reference=cfg.pop('auto_register'), module_type=module_type)

        use_legacy = is_legacy_model()
        if use_legacy or module_type not in [MindFormerModuleType.CONFIG, MindFormerModuleType.MODELS]:
            if 'type' not in cfg:
                raise KeyError(
                    '`cfg` or `default_args` must contain the key "type",'
                    'but got {}\n{}'.format(cfg, default_args)
                )

        if not (isinstance(default_args, dict) or default_args is None):
            raise TypeError(f'default_args must be a dict or None, but got {type(default_args)}')

        args = cfg.copy()
        if default_args is not None:
            for k, v in default_args.items():
                if k not in args:
                    args.setdefault(k, v)
                else:
                    args[k] = v

        if use_legacy:
            obj_type = args.pop('type')
        else:
            obj_type = cls.get_instance_type_from_cfg(args, module_type)

        if isinstance(obj_type, str):
            obj_cls = cls.get_cls(module_type, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise ValueError(f"Can't find class type {type} class name {obj_type} in class registry")

        try:
            if not use_legacy and module_type == MindFormerModuleType.MODELS:
                return obj_cls(default_args['config'])
            return obj_cls(**args)
        except Exception as e:
            raise type(e)(f'{obj_cls.__name__}: {e}')

    @classmethod
    def get_instance(cls, module_type=MindFormerModuleType.TOOLS, class_name=None, **kwargs):
        """
        Gets an instance of the class in the registry.

        Args:
            module_type (MindFormerModuleType, optional): Module type name of MindFormers.
                Default: ``MindFormerModuleType.TOOLS``.
            class_name (str, optional): Class name. Default: ``None``.
            kwargs (Any): Additional keyword arguments for constructing instances of the class.

        Returns:
            An instance of the class.

        Raises:
            ValueError: `class_name` cannot be ``None``.
            ValueError: Can't find class `class_name` of type `module_type` in the registry.
        """
        if class_name is None:
            raise ValueError("Class name cannot be None.")

        if isinstance(class_name, str):
            obj_cls = cls.get_cls(module_type, class_name)
        elif inspect.isclass(class_name):
            obj_cls = class_name
        else:
            raise ValueError(f"Can't find class type {type} class name {class_name} in class registry.")

        try:
            return obj_cls(**kwargs)
        except Exception as e:
            raise type(e)(f'{obj_cls.__name__}: {e}')

    @classmethod
    def auto_register(cls, class_reference: str, module_type=MindFormerModuleType.TOOLS):
        """
        Auto register function.

        Args:
            class_reference (str): The full name of the class to load.
            module_type (MindFormerModuleType.TOOLS): module type.
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
        _ = cls.register_cls(module_class, module_type=module_type, legacy=is_legacy_model())

    @classmethod
    def _add_class_name_prefix(cls, module_type, class_name, legacy=True):
        if not legacy and module_type in [MindFormerModuleType.MODELS, MindFormerModuleType.CONFIG]:
            class_name = NEW_CLASS_PREFIX + class_name
        return class_name
