# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Configuration."""

# pylint: disable=W0613
import inspect
import copy
import os
import re
from collections import deque
from functools import partial
from abc import ABCMeta, abstractmethod
from typing import List

import yaml

import mindspore.common.dtype as mstype
try:
    from mindspore._checkparam import Validator
    from mindspore._checkparam import Rel
except ImportError:
    import mindspore._checkparam as Validator
    import mindspore._checkparam as Rel
from mindspore.common.initializer import _INITIALIZER_ALIAS

from mindformers.tools import DictConfig, logger
from mindformers.experimental.distri_cores.utils import load_yaml, DictWithValueError


_SUPPORT_DTYPE_DICT = DictWithValueError(
    {"float16": mstype.float16, "float32": mstype.float32, "bfloat16": mstype.bfloat16}
)

_SUPPORT_INIT_METHOD = DictWithValueError(_INITIALIZER_ALIAS)


def config_to_str(cls, gap=2 * " "):
    """Return class attribute str for print."""
    attributes = vars(cls)
    print_str = "\n" + cls.__class__.__name__ + "\n"
    for name, val in attributes.items():
        new_str = str(val)
        new_str = new_str.replace("\n", "\n" + gap)
        print_str += f"{gap}{name}: {new_str}\n"

    return print_str


def dict_to_dictconfig(input_dict):
    """Convert dict to DictConfig."""
    if isinstance(input_dict, dict):
        for key, value in input_dict.items():
            input_dict[key] = dict_to_dictconfig(value)
        return DictConfig(**input_dict)
    return input_dict


class BaseConfig(metaclass=ABCMeta):
    """
    Base config class, which enables validator registration while attribute setting, and depended config registration.

    Notice:
        All dict attributes will be transformered to DictConfig Recursively.
    """

    _validation_func_dict = {}
    _depended_configs = {}  # {config_class: optional_flag}

    @abstractmethod
    def __init__(self):
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._validation_func_dict = {}
        cls._depended_configs = {}
        for base in cls.__bases__:
            if issubclass(base, BaseConfig):
                # pylint: disable=W0212
                cls._validation_func_dict.update(base._validation_func_dict)
                # pylint: disable=W0212
                cls._depended_configs.update(base._depended_configs)

        # check if config_name is set
        if not hasattr(cls, "config_name"):
            raise ValueError(f"Config {cls.__name__} should have a 'config_name' class attribute.")

    def __setattr__(self, name, value):
        """Set attribute with validation."""
        validator_func = self._validation_func_dict.get(name)
        if validator_func is not None:
            value = validator_func(self, value)
        if isinstance(value, dict):
            value = dict_to_dictconfig(value)
        super().__setattr__(name, value)

    def __str__(self):
        return config_to_str(self)

    def update_attrs(self, **kwargs):
        """
        Update the attributes of the object with the given key-value pairs. Dict attributes will be transformed
            to DictConfig recursively.

        Args:
            **kwargs: Key-value pairs where the key is the attribute name and the value is the new value.

        Returns:
            None
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def get_all_sub_configs(cls):
        """
        Returns a set of all subclasses of the given class, including subclasses of subclasses.

        Returns:
            visited_configs (set[BaseConfig]): A set containing all subclasses of the given class.

        """
        visited_configs = set(cls.__subclasses__())
        bfs_queue = deque(visited_configs)

        while bfs_queue:
            current_config = bfs_queue.popleft()
            for direct_sub_config in current_config.__subclasses__():
                if direct_sub_config not in visited_configs:
                    visited_configs.add(direct_sub_config)
                    bfs_queue.append(direct_sub_config)
        return visited_configs

    @classmethod
    def is_depended_config(cls, config_class, raise_error=False):
        """Check whether the config class is a depended config.

        Args:
            config_class (BaseConfig): The config class to be checked.
        """
        if not issubclass(config_class, BaseConfig):
            raise ValueError(f"{config_class} should be a subclass of BaseConfig.")
        if raise_error and config_class not in cls._depended_configs:
            raise ValueError(f"{config_class} is not a depended config of {cls}.")
        return config_class in cls._depended_configs

    @classmethod
    def _register_depended_config(cls, config_class, optional=False):
        """Register depended config class and add validation function for the depended config.

        Args:
            config_class (BaseConfig): The depended config class.
            optional (bool): Whether the depended config is optional.
        """
        if cls.is_depended_config(config_class):
            logger.warning(f"{config_class} is already a depended config of {cls}.")

        # add validation function for the depended config
        if optional:
            def validate_config(config_instance, config):
                if config is not None and not isinstance(config, config_class):
                    raise ValueError(f"{config} should be an instance of {config_class} or None.")
                return config
        else:
            def validate_config(config_instance, config):
                if not isinstance(config, config_class):
                    raise ValueError(f"{config} should be an instance of {config_class}.")
                return config

        cls._validation_func_dict[config_class.__name__] = validate_config
        cls._depended_configs[config_class] = optional

    @classmethod
    def _remove_depended_config(cls, config_class):
        """Remove depended config class and validation function for the depended config.

        Args:
            config_class (BaseConfig): The depended config class to be removed.

        Raises:
            ValueError: If config_config is not config.
            ValueError: If config_class is not a depended config of cls.
        """
        cls.is_depended_config(config_class, raise_error=True)
        # remove validation function for the depended config
        removed_class = cls._validation_func_dict.pop(config_class.__name__)
        removed_class_optional_flag = cls._depended_configs.pop(config_class)
        return removed_class, removed_class_optional_flag

    @classmethod
    def remove_depended_config(cls, config_class):
        """Remove depended config class(s) and validation function(s) for the depended config(s).

        Args:
            config_class (Union[BaseConfig, list[BaseConfig]]): The depended config
                class or a list of depended config classes.
        """
        if isinstance(config_class, list):
            for config in config_class:
                cls._remove_depended_config(config)
        else:
            cls._remove_depended_config(config_class)

    @classmethod
    def update_depended_config(cls, original_config, new_config=None, optional=None):
        """Update depended config class and validation function for the depended config.

        Args:
            original_config (BaseConfig): The original depended config class.
            new_config (BaseConfig, optional): The new depended config class. if this is set,
                the original depended config will be replaced by the new depended config. Default: None.
            optional (bool, optional): Whether the new depended config is optional. if this is set,
                the optional flag of the depended config will be updated. Default: None.
        """
        cls.is_depended_config(original_config, raise_error=True)

        if new_config is None and optional is None:
            logger.warning("No new depended config or optional flag is set.")
        else:
            _, original_optional_flag = cls._remove_depended_config(original_config)
            optional_flag = optional if optional is not None else original_optional_flag
            new_config = new_config if new_config is not None else original_config
            cls._register_depended_config(new_config, optional_flag)

    @classmethod
    def get_depended_configs(cls):
        """get depended configs

        Returns:
            Dict[BaseConfig, bool]: The depended config class and optional flag.
        """
        return copy.deepcopy(cls._depended_configs)

    @classmethod
    def register_depended_config(cls, config_class, optional=False):
        """Register depended config class(s) and add validation function(s) for the depended config(s).

        Args:
            config_class (Union[BaseConfig, list[BaseConfig]]): The depended config
                class or a list of depended config classes.
            optional (Union[bool, list[bool]], optional): Whether the depended config is optional.
                if is a list, the length of the list should be the same as the config_class. Default: False.
        """
        if isinstance(config_class, list):
            optional_flags = optional if isinstance(optional, list) else [optional] * len(config_class)
            if len(config_class) != len(optional_flags):
                raise ValueError("The length of config_class and optional should be the same.")
            for idx, config in enumerate(config_class):
                cls._register_depended_config(config, optional=optional_flags[idx])
        else:
            cls._register_depended_config(config_class, optional=optional)

    @classmethod
    def validator(cls, name):
        """Validator decorator, which registers validation function for attribute.
        Once the validation function is registered, the attribute will be validated while __setattr__ is called.

        Args:
            name (str): The name of the attribute to be validated.

        Returns:
            decorator: The decorator function.
        """

        def decorator(func):
            cls._validation_func_dict[name] = func
            return func

        return decorator

    def get_extra_params(self):
        """Get extra params that are not defined in the __init__ methods.

        Returns:
            dict: Extra params.
        """
        known_params = inspect.signature(self.__class__).parameters
        return {k: v for k, v in self.__dict__.items() if k not in known_params}

    def get_needed_params_for_class(self, a_class):
        """
        Returns a dictionary of the needed parameters for a given class.

        Args:
            a_class: The class for which the needed parameters are to be retrieved.

        Returns:
            A dictionary containing the needed parameters and their corresponding values from the current instance.
        """
        needed_parameters = inspect.signature(a_class).parameters.keys()
        return {k: v for k, v in self.__dict__.items() if k in needed_parameters}


class AllConfig(BaseConfig):
    """A Config that contains all other configs, which will be used in init_configs methods as the default config.

    Args:
        **kwargs: Other arguments.
    """

    # set config name for identifying while using init_configs methods
    config_name = "all_config"

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__()
        self.update_attrs(**kwargs)

    @classmethod
    def register_all_known_configs(cls):
        """Register all known configs. Known configs are the subclasses of BaseConfig.

        Notice:
            - This method should be called after all known configs are defined.
        """
        all_configs = BaseConfig.get_all_sub_configs()
        all_configs.remove(cls)

        cls.register_depended_config(list(all_configs), optional=True)


def build_dependency_graph_of_configs(config_classes):
    """
    Builds a dependency graph of configuration classes based on their dependencies.

    Args:
        config_classes (list[BaseConfig]): A list of configuration classes.

    Returns:
        dependency_graph dict[BaseConfig, list[BaseConfig]]: A dictionary representing the dependency graph,
        where the keys are configuration classes and the values are lists of configuration classes that
        they depend on.

    Raises:
        ValueError: If a configuration class is not a subclass of BaseConfig.
        ValueError: If a configuration class does not have the 'config_name' attribute set.
        ValueError: If a cycle is detected in the configuration dependencies.

    """
    dependency_graph = {}
    for config_class in config_classes:
        # check if config_class is already in dependency_graph
        if config_class in dependency_graph:
            continue
        # check class
        if not issubclass(config_class, BaseConfig):
            raise ValueError(f"{config_class} should be a subclass of BaseConfig.")
        # check if config_name is set
        if not hasattr(config_class, "config_name"):
            raise ValueError("You should set config_name for config class before using init_configs methods.")

        depended_configs_dict = config_class.get_depended_configs()
        depended_configs = []

        # get depended configs
        for depended_config, optional_flag in depended_configs_dict.items():
            # check cycle dependency
            if depended_config in dependency_graph and config_class in dependency_graph[depended_config]:
                raise ValueError(
                    "Cycle detected in configuration dependencies:"
                    + f"{config_class} and {depended_config} form a loop."
                )
            # filter out optional depended config which is not passed in
            if not optional_flag or depended_config in config_classes:
                depended_configs.append(depended_config)
                if depended_config not in config_classes:
                    config_classes.append(depended_config)
                    logger.warning(
                        f"Will initialize config {depended_config.config_name}"
                        + f"since it is required by {config_class.config_name}."
                    )

        dependency_graph[config_class] = depended_configs
    return dependency_graph


# pylint: disable=W0102
def init_configs_from_dict(raw_dict: dict, config_classes=None):
    """
    Initialize config class from configuration dictionary.

    Args:
        raw_dict (dict): Configuration dictionary.
        config_classes (Union[list[BaseConfig], None]): Config classes to be initialized. When no config class
            is passed in, all known configs will be initialized as optional config of AllConfig. Default: None

    Returns:
        Union[list[BaseConfig], AllConfig]: Initialized config instances, when no config class is passed in,
            AllConfig will be returned.
    """
    # check if no config class is passed in
    no_passed_in_configs = config_classes is None

    # when no config class is passed in, all known configs will be initialized as optional config of AllConfig
    if no_passed_in_configs:
        config_classes = [AllConfig]
        # register here to ensure all known configs can be reached
        AllConfig.register_all_known_configs()
        # read configs to be initialized directly from raw_dict
        known_configs = AllConfig.get_depended_configs().keys()
        for known_config in known_configs:
            if known_config.config_name in raw_dict:
                config_classes.append(known_config)
        # make sure all_config is in raw_dict
        if AllConfig.config_name not in raw_dict:
            raw_dict[AllConfig.config_name] = {}
    else:
        returned_config_names = [config_class.config_name for config_class in config_classes]
        if AllConfig in config_classes:
            AllConfig.register_all_known_configs()

    # construct dependency graph
    dependency_graph = build_dependency_graph_of_configs(config_classes)

    # topological sort
    visited = {config_class: False for config_class in config_classes}
    config_class_initialization_stack = deque()

    def dfs(config_class):
        visited[config_class] = True
        for dependency in dependency_graph[config_class]:
            if not visited[dependency]:
                dfs(dependency)
        config_class_initialization_stack.append(config_class)

    for config_class in config_classes:
        if not visited[config_class]:
            dfs(config_class)

    # initialize configs
    initialized_configs = {}
    while config_class_initialization_stack:
        config_class = config_class_initialization_stack.popleft()
        if config_class.config_name not in raw_dict:
            raise ValueError(f"Config {config_class.config_name} not found.")
        kwargs = raw_dict[config_class.config_name]
        depened_config_instances = {
            depended_config.config_name: initialized_configs[depended_config.config_name]
            for depended_config in dependency_graph[config_class]
        }
        kwargs.update(depened_config_instances)
        config_instance = config_class(**kwargs)
        initialized_configs[config_class.config_name] = config_instance
        logger.warning(f"Initialized config {config_class.config_name}:")
        logger.warning(config_instance)

    # if no passed in configs, add all other parameters to AllConfig as dict config
    if no_passed_in_configs:
        for config_name in raw_dict.keys():
            if config_name not in initialized_configs:
                setattr(initialized_configs[AllConfig.config_name], config_name, raw_dict[config_name])
        return initialized_configs[AllConfig.config_name]

    # return in order if config classes are passed in
    return [initialized_configs[config_name] for config_name in returned_config_names]


# pylint: disable=W0102
def init_configs_from_yaml(file_path: str, config_classes=None, **kwargs):
    """Initialize config class from configuration yaml file.

    Args:
        file_path (str): configuration yaml file.
        config_classes (Union[list[BaseConfig], None]): Config classes to be initialized. When no config class
            is passed in, all known configs will be initialized as optional config of AllConfig. Default: None
        kwargs (dict): extra arguments.

    Returns:
        Union[list[BaseConfig], AllConfig]: Initialized config instances, when no config class is passed in,
            AllConfig will be returned.
    """
    if not isinstance(file_path, str):
        raise ValueError("file_path should be a string.")
    if not file_path.endswith("yaml") and not file_path.endswith("yml"):
        raise ValueError("file_path should be a yaml file.")
    filepath = os.path.realpath(file_path)
    with open(filepath, encoding="utf-8") as fp:
        raw_dict = load_yaml(fp, yaml_loader=yaml.FullLoader)

    raw_dict.update(kwargs)

    return init_configs_from_dict(raw_dict, config_classes)


class GeneralConfig(BaseConfig):
    """A General Config

    Args:
        **kwargs: Arbitrary keyword arguments.
    """

    # set config name for identifying while using init_configs methods
    config_name = "general_config"

    def __init__(self, **kwargs):
        super().__init__()
        self.update_attrs(**kwargs)


class LoraConfig(BaseConfig):
    r"""LoRA config class.

    Args:
        use_lora (bool): Apply LoRA to the pretrain model. Default: False.
        lora_rank (int): The dimension for LoRA modules. Default: 8.
        lora_alpha (int): The alpha parameter for LoRA scaling. Default: 32.
        lora_dropout (float): the dropout rate for LoRA. Default: 0.0.
        target_cells (list[dict]): The names of the cells to build LoRA modules. If 'use_lora' is
            True, this argument should at least contains a dict with the key 'targets_cells' and
            the value of names of the cells to apply LoRA. In addition, if you want to set special
            rank or alpha for cells in target_cells, you can add dict to the list.
            For example:
            case 1:
                target_cells = [
                  {'target_cells':[
                      '.*.qkv_proj'
                  ]},
              ]
            In this case, cells which name end with '.qkv_proj' will be applied LoRA.

            case 2:
                target_cells = [
                  {'target_cells':[
                      'backbone.layers.layers.0.attention.qkv_proj'
                  ]},
              ]
            In this case, the cell 'backbone.layers.layers.0.attention.qkv_proj' will be applied LoRA.

            case 3:
                [
                  {'target_cells':[
                      '.*.qkv_proj',
                  ]},
                  {'cell':'backbone.layers.layers.0.attention.qkv_proj', 'rank':4, 'alpha':16},
              ]
            In this case, cells which name end with '.qkv_proj' will be applied LoRA. In addition, the rank
            and alpha of the cell 'backbone.layers.layers.0.attention.qkv_proj' is 4 and 32, the rank and
            alpha of other cells are set to 'lora_rank' and 'lora_alpha'.
    """
    config_name = "lora_config"

    def __init__(
            self,
            use_lora: bool = False,
            lora_rank: int = 8,
            lora_alpha: int = 32,
            lora_dropout: float = 0.0,
            target_cells: List = None,
            **kwargs,
    ):
        super().__init__()
        self.use_lora = use_lora
        if use_lora:
            self.lora_rank = lora_rank
            self.lora_alpha = lora_alpha
            self.lora_dropout = lora_dropout
            self.target_cells = target_cells
            self.lora_module = None

            self.update_attrs(**kwargs)


@LoraConfig.validator("use_lora")
def validate_use_lora(config_instance, use_lora):
    """Validate lora_rank."""
    Validator.check_bool(use_lora, "use_lora")
    return use_lora


@LoraConfig.validator("lora_rank")
def validate_lora_rank(config_instance, lora_rank):
    """Validate lora_rank."""
    Validator.check_positive_int(lora_rank, "lora_rank")
    return lora_rank


@LoraConfig.validator("lora_alpha")
def validate_lora_alpha(config_instance, lora_alpha):
    """Validate lora_alpha."""
    Validator.check_positive_int(lora_alpha, "lora_alpha")
    return lora_alpha


@LoraConfig.validator("lora_dropout")
def validate_lora_dropout(config_instance, lora_dropout):
    """Validate lora_dropout."""
    Validator.check_non_negative_float(lora_dropout, "lora_dropout")
    return lora_dropout


@LoraConfig.validator("target_cells")
def validate_target_cells(config_instance, target_cells):
    """Validate target_cells."""
    Validator.check_value_type("target_cells", target_cells, list)
    if not target_cells:
        raise ValueError("'target_cells' cannot not be empty.")

    # valid target_cells
    target_cells_defined = False
    for item in target_cells:
        if 'target_cells' in item.keys():
            if target_cells_defined:
                raise ValueError("'target_cells' cannot not be defined more than once.")
            target_cells_defined = True
            Validator.check_value_type("target_cells", item['target_cells'], list)
            target_cells_lst = item['target_cells']
            if not target_cells_lst:
                raise ValueError("for 'target_cells', the list of target_cells name must be set.")
    if not target_cells_defined:
        raise ValueError("for 'target_cells', the list of target_cells name must be set.")

    def _check_in_target_cells(cell_name):
        target_cell_found = False
        for target_key in target_cells_lst:
            match = re.match(target_key, cell_name)
            if match is not None and match.group() == cell_name:
                return target_key
        return target_cell_found

    # valid rank and alpha for specific cells
    specific_lora_cell = []
    for item in target_cells:
        if 'cell' in item.keys():
            cell_name = item['cell']
            if not _check_in_target_cells(cell_name):
                raise ValueError(
                    f"The cell need to set rank or alpha should be in the range defined by target_cells, but got name "
                    f"'{cell_name}'.")
            specific_lora_cell.append(item)
    return target_cells_lst, specific_lora_cell


class TrainingConfig(BaseConfig):
    """
    Training config.

    Args:
        epochs (int): Epochs number for training. Default: 1.
        batch_size (int): Batch size for training. Default: 1.
    """

    # set config name for identifying while using init_configs methods
    config_name = "training_config"

    def __init__(self, epochs: int = 1, batch_size: int = 1, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size

        self.update_attrs(**kwargs)


@TrainingConfig.validator("epochs")
def validate_epochs(config_instance, epochs):
    """Validate epochs"""
    Validator.check_positive_int(epochs, "epochs")
    return epochs


@TrainingConfig.validator("batch_size")
def validate_batch_size(config_instance, batch_size):
    """Validate batch_size"""
    Validator.check_positive_int(batch_size, "batch_size")
    return batch_size


class DatasetConfig(BaseConfig):
    """
    Dataset config class.

    Args:
        dataset_dir (str): Dataset file directory. Default: './dataset'.
        shuffle (Union[bool, None], optional): Shuffle dataset. Default: None.
    """

    # set config name for identifying while using init_configs methods
    config_name = "dataset_config"

    def __init__(self, dataset_dir: str = "./dataset", shuffle: bool = None, **kwargs):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.shuffle = shuffle

        self.update_attrs(**kwargs)


@DatasetConfig.validator("dataset_dir")
def validate_dataset_dir(config_instance, dataset_dir):
    """Validate dataset_dir."""
    Validator.check_value_type("dataset_dir", dataset_dir, [str])
    return dataset_dir


@DatasetConfig.validator("shuffle")
def validate_shuffle(config_instance, shuffle):
    """Validate shuffle."""
    Validator.check_value_type("shuffle", shuffle, [bool, type(None)])
    return shuffle


class ModelParallelConfig(BaseConfig):
    """
    Parallel config class.

    Args:
        tensor_parallel (int): Dimensionality of tensor parallel. Default: 1.
        pipeline_stage (int): Number of stages when using pipeline parallel. Default: 1.
        context_parallel (int): Dimensionality of context parallel. Default: 1.
        expert_parallel (int): Dimensionality of expert parallel. Default: 1.
        micro_batch_num (int): Number of micro batch when using pipeline parallel. Default: 1.
        use_sequence_parallel (bool): Enable sequence parallel. Default: False.
        recv_dtype (bool): Communication data type of p2p communication when using pipeline
            parallel. Default: 'float32'.
        use_zero3 (Union[bool, None]): Enable zero3 optimization. Default: None.
        gradient_accumulation_fusion (bool): Enable gradient accumulation
            during linear backward execution. Default: False.
    """

    # set config name for identifying while using init_configs methods
    config_name = "parallel_config"

    def __init__(
            self,
            tensor_parallel: int = 1,
            pipeline_stage: int = 1,
            context_parallel: int = 1,
            expert_parallel: int = 1,
            micro_batch_num: int = 1,
            use_sequence_parallel: bool = False,
            recv_dtype: str = "float32",
            zero_level: bool = None,
            gradient_accumulation_fusion: bool = False,
            **kwargs,
    ):
        super(ModelParallelConfig, self).__init__()
        self.tensor_parallel = tensor_parallel
        self.pipeline_stage = pipeline_stage
        self.context_parallel = context_parallel
        self.expert_parallel = expert_parallel
        self.micro_batch_num = micro_batch_num
        self.use_sequence_parallel = use_sequence_parallel
        self.recv_dtype = recv_dtype
        self.zero_level = zero_level
        self.gradient_accumulation_fusion = gradient_accumulation_fusion

        self.update_attrs(**kwargs)


@ModelParallelConfig.validator("tensor_parallel")
def validate_tensor_parallel(config_instance, tensor_parallel):
    """Validate tensor_parallel."""
    Validator.check_positive_int(tensor_parallel, "tensor_parallel")
    return tensor_parallel


@ModelParallelConfig.validator("pipeline_stage")
def validate_pipeline_stage(config_instance, pipeline_stage):
    """Validate pipeline_stage."""
    Validator.check_positive_int(pipeline_stage, "pipeline_stage")
    return pipeline_stage


@ModelParallelConfig.validator("context_parallel")
def validate_context_parallel(config_instance, context_parallel):
    """Validate context_parallel."""
    Validator.check_positive_int(context_parallel, "context_parallel")
    return context_parallel


@ModelParallelConfig.validator("expert_parallel")
def validate_expert_parallel(config_instance, expert_parallel):
    """Validate expert_parallel."""
    Validator.check_positive_int(expert_parallel, "expert_parallel")
    return expert_parallel


@ModelParallelConfig.validator("micro_batch_num")
def validate_micro_batch_num(config_instance, micro_batch_num):
    """Validate micro_batch_num."""
    Validator.check_positive_int(micro_batch_num, "micro_batch_num")
    return micro_batch_num


@ModelParallelConfig.validator("use_sequence_parallel")
def validate_use_sequence_parallel(config_instance, use_sequence_parallel):
    """Validate use_sequence_parallel."""
    Validator.check_bool(use_sequence_parallel, "use_sequence_parallel")
    return use_sequence_parallel


@ModelParallelConfig.validator("recv_dtype")
def validate_recv_dtype(config_instance, recv_dtype):
    """Validate recv_dtype."""
    return _SUPPORT_DTYPE_DICT[recv_dtype]


@ModelParallelConfig.validator("use_zero3")
def validate_use_zero3(config_instance, use_zero3):
    """Validate use_zero3."""
    Validator.check_value_type("use_zero3", use_zero3, [bool, type(None)])


@ModelParallelConfig.validator("gradient_accumulation_fusion")
def validate_gradient_accumulation_fusion(config_instance, gradient_accumulation_fusion):
    """Validate gradient_accumulation_fusion."""
    Validator.check_bool(gradient_accumulation_fusion, "gradient_accumulation_fusion")
    return gradient_accumulation_fusion


class TransformerConfig(BaseConfig):
    r"""
    Model config class.

    Args:
        vocab_size (int): Vocabulary size.
        num_layers (int): Number of model layers.
        num_heads (int): Number of heads for MultiHeadAttention.
        hidden_size (int): Dimensionality of the encoder layers.
        ffn_hidden_size (int): Dimensionality the FeedForward block project to.
        parallel_config (ParallelConfig): Parallel config.
        lora_config (LoraConfig): Lora config.
        attention_type (str): Attention type. Default: 'self_attn'.
        use_gqa (bool): Enable group query attention. Default: False.
        kv_num_heads (int): Number of heads for key and value when using group query attention.
            Default: 32.
        qkv_has_bias (bool): Linears apply on query, key and value in Attention block has bias
            parameter. Default: True.
        out_proj_has_bias (bool): Linear applies on output of core attention block has bias
            parameter. Default: True.
        apply_query_key_layer_scaling (bool): Apply query key scaling in core attention block.
            Default: False.
        use_flash_attention (bool): Enable flash attention. Default: False.
        mask_func_type (str): Attention mask compute method. Default: 'attn_mask_add'.
        mlp_has_bis (bool): Linears in MLP block have bias parameters. Default: True.
        mlp_has_gate (bool): Apply gating in MLP block. Default: False.
        hidden_act (str): Activation used in MLP block. Default: 'gelu'.
        normalization (str): Normalization used in transformerlayer block. Default: 'LayerNorm'.
        layernorm_epsilon (float): Epsilon of normalization. Default: 1.e-5.
        apply_residual_connection_post_norm (bool): Apply residual connection after normalization.
            Default: False.
        residual_connection_dtype (str): Compute data type of residual connection. Default: 'float32'.
        param_init_dtype (str): Parameter initialize data type. Default: 'float32'.
        compute_dtype (str): Compute data type of linear module. Default: 'float16'.
        softmax_compute_dtype (str): Compute data type of softmax layer. Default: 'float32'.
        hidden_dropout_rate (float): Dropout rate for output of attention block and mlp block in transformerlayer.
            Default: 0.0.
        attention_dropout_rate (float): Dropout rate for attention socre. Default: 0.0.
        num_experts (Optional[int], None): Number of experts. Default: None.
        share_embedding_weight (bool): Share embedding table between embedding layer and llm head. Default: False.
        recompute_method (Optional[str], None): Recompute method. Default: None.
        recompute_num_layers (Optional[int], None): Number of layers to recompute. Default: None.
        recompute_granularity (Optional[str], None): Recompute granularity. Default: None.
    """

    # set config name for identifying while using init_configs methods
    config_name = "model_config"

    def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            num_heads: int,
            hidden_size: int,
            ffn_hidden_size: int,
            parallel_config: ModelParallelConfig,
            lora_config: LoraConfig = LoraConfig(),
            attention_type: str = "self_attn",
            use_gqa: bool = False,
            kv_num_heads: int = 32,
            qkv_has_bias: bool = True,
            out_proj_has_bias: bool = True,
            apply_query_key_layer_scaling: bool = False,
            use_flash_attention: bool = False,
            fa_config=None,
            mask_func_type: str = "attn_mask_add",
            mlp_has_bias: bool = True,
            mlp_has_gate: bool = False,
            hidden_act: str = "gelu",
            normalization: str = "LayerNorm",
            layernorm_epsilon: float = 1.0e-5,
            apply_residual_connection_post_norm: bool = False,
            residual_connection_dtype: str = "float32",
            param_init_dtype: str = "float32",
            compute_dtype: str = "float16",
            softmax_compute_dtype: str = "float32",
            init_method: str = 'normal',
            bias_init: str = 'zeros',
            hidden_dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            num_experts: int = None,
            share_embedding_weight: bool = False,
            recompute_method: str = None,
            recompute_num_layers: int = None,
            recompute_granularity: str = None,
            dataset_config: DatasetConfig = None,
            **kwargs,
    ):
        super(TransformerConfig, self).__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.parallel_config = parallel_config
        self.lora_config = lora_config
        self.attention_type = attention_type
        self.use_gqa = use_gqa
        self.kv_num_heads = kv_num_heads
        self.qkv_has_bias = qkv_has_bias
        self.out_proj_has_bias = out_proj_has_bias
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.use_flash_attention = use_flash_attention
        self.fa_config = fa_config
        self.fa_config = fa_config
        self.mask_func_type = mask_func_type
        self.mlp_has_bias = mlp_has_bias
        self.mlp_has_gate = mlp_has_gate
        self.hidden_act = hidden_act
        self.normalization = normalization
        self.layernorm_epsilon = layernorm_epsilon
        self.apply_residual_connection_post_norm = apply_residual_connection_post_norm
        self.residual_connection_dtype = residual_connection_dtype
        self.param_init_dtype = param_init_dtype
        self.compute_dtype = compute_dtype
        self.softmax_compute_dtype = softmax_compute_dtype
        self.init_method = init_method
        self.bias_init = bias_init
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.num_experts = num_experts
        self.share_embedding_weight = share_embedding_weight
        self.recompute_method = recompute_method
        self.recompute_num_layers = recompute_num_layers
        self.recompute_granularity = recompute_granularity
        self.dataset_config = dataset_config

        if "recompute_activations" in kwargs:
            if kwargs["recompute_activations"]:
                self.recompute_granularity = "selective"
            kwargs.pop("recompute_activations")

        self.update_attrs(**kwargs)

    def update_lora_config(self, cell_name):
        lora_module = self.lora_config.lora_module
        self.lora_config.lora_module = None if lora_module is None else lora_module.get(cell_name, None)


TransformerConfig.register_depended_config([ModelParallelConfig,
                                            LoraConfig,
                                            DatasetConfig],
                                           optional=[False, True, True])


@TransformerConfig.validator("vocab_size")
def validate_vocab_size(config_instance, vocab_size):
    """Validate vocab_size."""
    Validator.check_positive_int(vocab_size, "vocab_size")
    return vocab_size


@TransformerConfig.validator("num_layers")
def validate_num_layers(config_instance, num_layers):
    """Validate num_layers."""
    Validator.check_positive_int(num_layers, "num_layers")
    return num_layers


@TransformerConfig.validator("num_heads")
def validate_num_heads(config_instance, num_heads):
    """Validate num_heads."""
    Validator.check_positive_int(num_heads, "num_heads")
    return num_heads


@TransformerConfig.validator("hidden_size")
def validate_hidden_size(config_instance, hidden_size):
    """Validate hidden_size."""
    Validator.check_positive_int(hidden_size, "hidden_size")
    return hidden_size


@TransformerConfig.validator("ffn_hidden_size")
def validate_ffn_hidden_size(config_instance, ffn_hidden_size):
    """Validate ffn_hidden_size."""
    Validator.check_positive_int(ffn_hidden_size, "ffn_hidden_size")
    return ffn_hidden_size


@TransformerConfig.validator("attention_type")
def validate_attention_type(config_instance, attention_type):
    """Validate attention_type."""
    Validator.check_value_type("attention_type", attention_type, [str])
    return attention_type


@TransformerConfig.validator("use_gqa")
def validate_use_gqa(config_instance, use_gqa):
    """Validate use_gqa."""
    Validator.check_bool(use_gqa, "use_gqa")
    return use_gqa


@TransformerConfig.validator("kv_num_heads")
def validate_kv_num_heads(config_instance, kv_num_heads):
    """Validate kv_num_heads."""
    Validator.check_positive_int(kv_num_heads, "kv_num_heads")
    return kv_num_heads


@TransformerConfig.validator("qkv_has_bias")
def validate_qkv_has_bias(config_instance, qkv_has_bias):
    """Validate qkv_has_bias."""
    Validator.check_bool(qkv_has_bias, "qkv_has_bias")
    return qkv_has_bias


@TransformerConfig.validator("out_proj_has_bias")
def validate_out_proj_has_bias(config_instance, out_proj_has_bias):
    """Validate out_proj_has_bias."""
    Validator.check_bool(out_proj_has_bias, "out_proj_has_bias")
    return out_proj_has_bias


@TransformerConfig.validator("apply_query_key_layer_scaling")
def validate_apply_query_key_layer_scaling(config_instance, apply_query_key_layer_scaling):
    """Validate apply_query_key_layer_scaling."""
    Validator.check_bool(apply_query_key_layer_scaling, "apply_query_key_layer_scaling")
    return apply_query_key_layer_scaling


@TransformerConfig.validator("use_flash_attention")
def validate_use_flash_attention(config_instance, use_flash_attention):
    """Validate use_flash_attention."""
    Validator.check_bool(use_flash_attention, "use_flash_attention")
    return use_flash_attention


@TransformerConfig.validator("fa_config")
def validate_fa_config(config_instance, fa_config):
    """Validate fa_config."""
    if fa_config is not None:
        if not isinstance(fa_config, dict):
            raise ValueError("fa_config should be a dict.")

        def _check_sparse_mode(sparse_mode):
            support_sparse_mode = (0, 1, 2, 3, 4)
            if sparse_mode not in support_sparse_mode:
                raise NotImplementedError(
                    "For flash attention, sparse_mode only support" "[0, 1, 2, 3, 4] for now, but got {}".format(
                        str(sparse_mode)
                    )
                )

        args_and_check_map = {
            "keep_prob": partial(
                Validator.check_float_range, lower_limit=0, upper_limit=1, rel=Rel.INC_BOTH, arg_name="keep_prob"
            ),
            "pre_tokens": partial(
                Validator.check_int_range,
                lower_limit=-2147483647,
                upper_limit=2147483647,
                rel=Rel.INC_BOTH,
                arg_name="pre_tokens",
            ),
            "next_tokens": partial(
                Validator.check_int_range,
                lower_limit=-2147483647,
                upper_limit=2147483647,
                rel=Rel.INC_BOTH,
                arg_name="next_tokens",
            ),
            "input_layout": partial(Validator.check_string, valid_values=("BNSD"), arg_name="input_layout"),
            "sparse_mode": _check_sparse_mode,
        }
        for arg_name, value in fa_config.items():
            if arg_name not in args_and_check_map.keys():
                raise ValueError(
                    "For FAConfig, only `keep_prob`, `pre_tokens`, `next_tokens`, `input_layout`, "
                    "and `sparse_mode` are configuable, but got {}".format(arg_name)
                )
            args_and_check_map[arg_name](value)
    return fa_config


@TransformerConfig.validator("mask_func_type")
def validate_mask_func_type(config_instance, mask_func_type):
    """Validate mask_func_type."""
    Validator.check_value_type("mask_func_type", mask_func_type, [str])
    return mask_func_type


@TransformerConfig.validator("mlp_has_bias")
def validate_mlp_has_bias(config_instance, mlp_has_bias):
    """Validate mlp_has_bias."""
    Validator.check_bool(mlp_has_bias, "mlp_has_bias")
    return mlp_has_bias


@TransformerConfig.validator("mlp_has_gate")
def validate_mlp_has_gate(config_instance, mlp_has_gate):
    """Validate mlp_has_gate."""
    Validator.check_bool(mlp_has_gate, "mlp_has_gate")
    return mlp_has_gate


@TransformerConfig.validator("hidden_act")
def validate_hidden_act(config_instance, hidden_act):
    """Validate hidden_act."""
    Validator.check_value_type("hidden_act", hidden_act, [str])
    return hidden_act


@TransformerConfig.validator("normalization")
def validate_normalization(config_instance, normalization):
    """Validate normalization."""
    Validator.check_value_type("normalization", normalization, [str])
    return normalization


@TransformerConfig.validator("layernorm_epsilon")
def validate_layernorm_epsilon(config_instance, layernorm_epsilon):
    """Validate layernorm_epsilon."""
    Validator.check_positive_float(layernorm_epsilon, "layernorm_epsilon")
    return layernorm_epsilon


@TransformerConfig.validator("apply_residual_connection_post_norm")
def validate_apply_residual_connection_post_norm(config_instance, apply_residual_connection_post_norm):
    """Validate apply_residual_connection_post_norm."""
    Validator.check_bool(apply_residual_connection_post_norm, "apply_residual_connection_post_norm")
    return apply_residual_connection_post_norm


@TransformerConfig.validator("residual_connection_dtype")
def validate_residual_connection_dtype(config_instance, residual_connection_dtype):
    """Validate residual_connection_dtype."""
    return _SUPPORT_DTYPE_DICT[residual_connection_dtype]


@TransformerConfig.validator("param_init_dtype")
def validate_param_init_dtype(config_instance, param_init_dtype):
    """Validate param_init_dtype."""
    return _SUPPORT_DTYPE_DICT[param_init_dtype]


@TransformerConfig.validator("compute_dtype")
def validate_compute_dtype(config_instance, compute_dtype):
    """Validate compute_dtype."""
    return _SUPPORT_DTYPE_DICT[compute_dtype]


@TransformerConfig.validator("softmax_compute_dtype")
def validate_softmax_compute_dtype(config_instance, softmax_compute_dtype):
    """Validate softmax_compute_dtype."""
    return _SUPPORT_DTYPE_DICT[softmax_compute_dtype]


@TransformerConfig.validator("hidden_dropout_rate")
def validate_hidden_dropout_rate(config_instance, hidden_dropout_rate):
    """Validate hidden_dropout_rate."""
    Validator.check_float_range(hidden_dropout_rate, 0, 1, Rel.INC_BOTH, "hidden_dropout_rate")
    return hidden_dropout_rate


@TransformerConfig.validator("attention_dropout_rate")
def validate_attention_dropout_rate(config_instance, attention_dropout_rate):
    """Validate attention_dropout_rate."""
    Validator.check_float_range(attention_dropout_rate, 0, 1, Rel.INC_BOTH, "attention_dropout_rate")
    return attention_dropout_rate


@TransformerConfig.validator("num_experts")
def validate_num_experts(config_instance, num_experts):
    """Validate num_experts."""
    Validator.check_value_type("num_experts", num_experts, [int, type(None)])
    return num_experts


@TransformerConfig.validator("share_embedding_weight")
def validate_share_embedding_weight(config_instance, share_embedding_weight):
    """Validate share_embedding_weight."""
    Validator.check_bool(share_embedding_weight, "share_embedding_weight")
    return share_embedding_weight


@TransformerConfig.validator("recompute_method")
def validate_recompute_method(config_instance, recompute_method):
    if recompute_method is not None:
        Validator.check_string(recompute_method, ["uniform", "block"], "recompute_method")
    return recompute_method


@TransformerConfig.validator("recompute_num_layers")
def validate_recompute_num_layers(config_instance, recompute_num_layers):
    """Validate recompute_num_layers."""
    if recompute_num_layers is not None:
        Validator.check_int_range(
            recompute_num_layers, 1, config_instance.num_layers, Rel.INC_BOTH, "recompute_num_layers"
        )
        if config_instance.recompute_method is None:
            logger.warning("recompute_method, recompute_num_layers should be set together.")
    return recompute_num_layers


@TransformerConfig.validator("recompute_granularity")
def validate_recompute_granularity(config_instance, recompute_granularity):
    """Validate recompute_granularity."""
    if recompute_granularity is not None:
        Validator.check_string(recompute_granularity, ["selective", "full"], "recompute_granularity")
        if recompute_granularity == "full":
            if config_instance.recompute_method is None:
                logger.warning("recompute_method should be set when recompute_granularity is set to 'full'.")
            if config_instance.recompute_num_layers is None:
                logger.warning("recompute_num_layers should be set when recompute_granularity is set to 'full'.")
    return recompute_granularity
