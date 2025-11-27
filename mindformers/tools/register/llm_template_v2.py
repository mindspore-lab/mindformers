# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Config and Template"""
import copy
from typing import Dict, List, Union, Any, Optional

from mindformers.tools.logger import logger
from .validate_types_and_ranges import validate_config_types_and_ranges


class Config:
    """
    A base class for applying structured configuration.

    This class serves as a blueprint for handling configuration data by providing methods
    to validate, update, and manage key-value pairs from config files or dictionaries. It
    supports configuration inputs as dictionaries, allowing for flexible configuration
    management in workflows.

    Attributes:
        _name (str): The name of the configuration. This can be set by subclasses to
                     differentiate between different types of configurations.
        _raise_error_for_unexpected_key (bool): If True, raises an error when encountering
                     unexpected keys in the input configuration. If False, ignores them.
        _support_none_input (bool): If True, allows the configuration input to be None.
                     If False, raises a ValueError when the input is empty.
        _required_keys (dict): A dictionary specifying required keys for each configuration section.
                     If any required key is missing, a KeyError is raised.
        _exclude_recursive_dict_key (list): A list of keys that should be excluded from recursive
                     dictionary processing. When a configuration contains nested dictionaries,
                     values corresponding to these keys will not be recursively updated but
                     directly assigned instead.
        _validation_rules (dict): A dictionary defining validation rules for configuration parameters.
                     Each key represents a parameter name, and each value is a dictionary containing:
                     - 'type': Expected data type (e.g., int, str, bool, or tuple of types)
                     - 'range': Allowed value range or custom validation function
                     - 'description': Description of the parameter
                     These rules are validated using validate_config_types_and_ranges function to
                     ensure parameters conform to expected types and value ranges.
    """
    _name: str = ""
    _raise_error_for_unexpected_key: bool = True
    _support_none_input: bool = True
    _required_keys: Dict[str, List[str]] = {}
    _exclude_recursive_dict_key: List[str] = []
    _validation_rules: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def apply(cls, config_key: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply the configuration dictionary to the class, with validation.

        Args:
            config_key (str): The key identifying the specific configuration section to apply.
                Used to look up required keys in the _required_keys dictionary.
            config (Optional[Dict[str, Any]]): The configuration to apply. Can be a dictionary or None.

        Returns:
            Dict[str, Any]: A dictionary containing the final configuration with default and updated values.

        Raises:
            TypeError: If the input config is not a dict or None.
            ValueError: If the config is empty when _support_none_input is False.
            KeyError: If a required key is missing or an unexpected key is found in the config.
        """
        if config is None:
            config = {}

        if not isinstance(config, dict):
            raise TypeError(f"The input config should be a dict, but get {type(config)}")

        if not config and not cls._support_none_input:
            raise ValueError(f"The config '{cls._name}' is empty. Please check the yaml file.")

        if not config:
            return cls._none_process()
        result = cls._initialize_result()
        return cls._update_value(result, config_key, config)

    @classmethod
    def _none_process(cls) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process when the input configuration is None or empty.

        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: Default configuration values.
        """
        logger.warning(f"The input config {cls._name} is empty.")
        return cls.default_value()

    @classmethod
    def _initialize_result(cls) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Initialize the result dictionary with default values.

        Returns:
            dict: A dictionary containing default configuration values.
        """
        return cls.default_value()

    @classmethod
    def _update_value(cls, result: Dict[str, Any], config_key: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the result dictionary with values from the input config.

        Args:
            result (dict): The initial result dictionary to be updated.
            config_key (str): The key identifying the specific configuration section.
            config (dict): The input configuration dictionary.

        Returns:
            dict: The updated configuration dictionary.

        Raises:
            KeyError: If required keys are missing from any configuration dictionary.
        """
        config = copy.deepcopy(config)
        for required_key in cls._required_keys.get(config_key, []):
            if required_key not in config:
                raise KeyError(f"The config '{cls._name}' is missing a required key: {required_key}.")

        for key, value in config.items():
            if isinstance(value, dict) and key not in cls._exclude_recursive_dict_key:
                if key not in result:
                    if cls._raise_error_for_unexpected_key:
                        raise KeyError(f"The config '{cls._name}' gets an unexpected key: {key}")
                    result[key] = {}
                current_result = result[key]
                # If current_result is None, initialize it as an empty dict for recursive update
                if current_result is None:
                    current_result = {}
                required_value = cls._update_value(current_result, key, value)
                result[key] = required_value
                continue

            if key in result.keys() or not cls._raise_error_for_unexpected_key:
                result[key] = value
            else:
                raise KeyError(f"The config '{cls._name}' gets an unexpected key: {key}")
        return result

    @classmethod
    def keys(cls) -> List[str]:
        """
        Get all non-private attribute names of the class.

        Returns:
            list: A list of attribute names defined in the class (excluding private ones).
        """
        return [k for k in cls.__dict__ if not k.startswith("_")]

    @classmethod
    def default_value(cls) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get the default values for all configuration attributes.

        Returns:
            dict: A dictionary mapping attribute names to their default values.
        """
        return {key: getattr(cls, key) for key in cls.keys()}

    @classmethod
    def validate_config(cls, config_dict: Dict[str, Any]) -> None:
        """
        Validate a configuration dictionary against predefined rules.

        Args:
            config_dict: Dictionary of configuration parameters

        Returns: None

        Raises:
            ValueError: If any parameter fails validation
        """

        if not cls._validation_rules:
            return

        results = {}
        errors = []

        for param_name, param_value in config_dict.items():
            if param_name in cls._validation_rules:
                rule = cls._validation_rules[param_name]

                try:
                    validate_config_types_and_ranges(
                        param_value=param_value,
                        param_type=rule["type"],
                        value_range=rule.get("range"),
                        param_name=param_name
                    )
                    results[param_name] = {"status": "valid", "value": param_value}
                except (TypeError, ValueError) as e:
                    results[param_name] = {"status": "invalid", "error": str(e)}
                    errors.append(f"{param_name}: {str(e)}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

        return

    @classmethod
    def name(cls) -> str:
        """
        Get the name of the configuration.

        Returns:
            str: The name of the configuration class.
        """
        return cls._name


class SpecConfig(Config):
    """
    A specialized configuration class that does not raise errors for unexpected keys.
    """
    _raise_error_for_unexpected_key: bool = False

    @classmethod
    def _initialize_result(cls) -> Dict[str, Any]:
        """
        Initialize an empty result dictionary.

        Returns:
            dict: An empty dictionary.
        """
        return {}


class ListConfig(Config):
    """
    A configuration class for handling list-based configurations.

    This class extends config the base class to handle configurations provided as lists.
    It supports updating configurations based on input lists and allows for specifying default values.

    It is especially useful for cases where configurations are structured as a list of dictionaries,
    each representing a configuration block.

    Attributes:
        _name (str): The name identifier for the configuration.
        _raise_error_for_unexpected_key (bool): If False, unexpected keys in the config will not raise errors.
        _support_none_input (bool): Determines if None input is supported without raising errors.
        _required_keys (dict): A dictionary specifying required keys for each configuration section.
    """
    _raise_error_for_unexpected_key: bool = False

    @classmethod
    def apply(cls, config_key: str, config: Optional[List[Dict[str, Any]]]) -> Union[
        Dict[str, Any], List[Dict[str, Any]]]:
        """
        Apply the list-based configuration to the class, with validation.

        Args:
            config_key (str): The key identifying the specific configuration section to apply.
            config (Union[list, None]): The configuration to apply, expected to be a list of dictionaries.
                If None is provided, the behavior depends on _support_none_input.

        Returns:
            list: A list containing the final configuration with default and updated values.

        Raises:
            TypeError: If the input config is not a list.
            ValueError: If the config is empty when _support_none_input is False.
        """
        if config is None:
            config = []

        if not isinstance(config, list):
            raise TypeError(f"The input config should be a list, but get {type(config)}")

        if not config and not cls._support_none_input:
            raise ValueError(f"The config '{cls._name}' is empty. Please check the yaml file.")

        if not config:
            return cls._none_process()
        result = cls._initialize_result()
        return cls._update_value_list(result, config_key, config)

    @classmethod
    def _update_value_list(cls, result: List[Dict[str, Any]], config_key: str,
                           config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update the list-based result with values from the input config list.

        Args:
            result (list): The initial result list to be updated.
            config_key (str): The key identifying the specific configuration section.
            config (list): The input configuration list, where each element is expected to be a dictionary.

        Returns:
            list: The updated list containing configurations.
        """
        config = copy.deepcopy(config)
        for value in config:
            if not isinstance(value, dict) or "type" not in value:
                raise ValueError(
                    f"Each config item must be a dict with 'type' key, but got {type(value)}"
                )
            res_tmp = {}
            res_tmp = cls._update_value(res_tmp, config_key, value)
            type_name = value["type"]
            if type_name in cls._types_to_index():
                result[cls._types_to_index()[type_name]] = res_tmp.copy()
            else:
                result.append(res_tmp.copy())
        return result

    @classmethod
    def default_value(cls) -> List[Any]:
        """
        Get the default values for all configuration attributes as a list.

        Returns:
            list: A list containing default configuration values.
        """
        return [getattr(cls, key) for key in cls.keys()]

    @classmethod
    def _types_to_index(cls) -> Dict[str, int]:
        """
        Create a mapping from type names to their indices in the default value list.

        Returns:
            dict: A dictionary mapping type names to indices.
        """
        dic = {}
        for i, value in enumerate(cls.default_value()):
            dic[value["type"]] = i
        return dic


class TrainingGeneralConfig(Config):
    """Training general configuration"""
    run_mode: Optional[str] = None
    output_dir: str = "./output"
    use_parallel: bool = False
    use_legacy: bool = False
    pretrained_model_dir: str = ""
    train_precision_sync: bool = False

    _name: str = "training_general_config"
    _validation_rules: Dict[str, Dict[str, Any]] = {
        "run_mode": {
            "type": str,
            "range": ["train", "finetune"],
            "description": ("Training mode selection, 'train' for training from scratch, "
                          "'finetune' for fine-tuning a pretrained model")
        },
        "output_dir": {
            "type": str,
            "range": None,
            "description": ("Directory path to save training outputs including "
                          "checkpoints, logs, and final model weights")
        },
        "use_parallel": {
            "type": bool,
            "range": None,
            "description": ("Enable distributed training across multiple devices "
                          "for faster training of large models")
        },
        "use_legacy": {
            "type": bool,
            "range": [False],
            "description": ("Legacy mode flag, only mcore architecture is supported, "
                          "do not enable legacy mode")
        },
        "pretrained_model_dir": {
            "type": (str, None),
            "range": None,
            "description": ("Path to the pretrained model directory for loading "
                          "initial weights and tokenizer")
        },
        "train_precision_sync": {
            "type": bool,
            "range": [True, False],
            "description": ("Enable deterministic computation for training reproducibility, "
                          "may impact training speed")
        },
    }


class InferGeneralConfig(Config):
    """Inference general configuration"""
    run_mode: Optional[str] = None
    output_dir: str = "./output"
    use_parallel: bool = False
    use_legacy: bool = False
    pretrained_model_dir: str = ""

    predict_batch_size: int = 1
    load_checkpoint: str = ''
    load_ckpt_format: str = 'safetensors'
    infer_seed: int = 1234
    infer_precision_sync: bool = False

    adapter_id: Optional[str] = None

    input_data: str = "Please input your question."

    _name: str = "infer_general_config"

    _validation_rules: Dict[str, Dict[str, Any]] = {
        "run_mode": {
            "type": str,
            "range": ["predict"],
            "description": "Inference mode selection, must be 'predict' for model inference tasks"
        },
        "pretrained_model_dir": {
            "type": (str, None),
            "range": None,
            "description": ("Path to the directory containing pretrained model weights "
                          "and configuration files")
        },
        "output_dir": {
            "type": str,
            "range": None,
            "description": ("Directory path to save inference outputs including "
                          "generated texts and logs")
        },
        "use_parallel": {
            "type": bool,
            "range": None,
            "description": ("Enable distributed inference across multiple devices "
                          "for faster text generation")
        },
        "use_legacy": {
            "type": bool,
            "range": [False],
            "description": ("Legacy mode flag, only mcore architecture is supported, "
                          "do not enable legacy mode")
        },
        "predict_batch_size": {
            "type": int,
            "range": (1, 1e32),
            "description": ("Number of input sequences processed simultaneously "
                          "during inference")
        },
        "load_checkpoint": {
            "type": (str, None),
            "range": None,
            "description": "File path or directory path to load model checkpoint weights"
        },
        "load_ckpt_format": {
            "type": str,
            "range": ['safetensors'],
            "description": ("Format of the checkpoint file to load, "
                          "e.g., 'safetensors' for secure tensor storage")
        },
        "infer_seed": {
            "type": int,
            "range": (0, 1e32),
            "description": "Random seed for generating reproducible inference results"
        },
        "infer_precision_sync": {
            "type": bool,
            "range": None,
            "description": ("Enable deterministic computation for inference reproducibility, "
                          "may impact inference speed")
        },
        "adapter_id": {
            "type": (str, None),
            "range": None,
            "description": ("Identifier for adapter modules (e.g., LoRA) "
                          "to be loaded with the base model")
        },
        "input_data": {
            "type": str,
            "range": None,
            "description": "Path to a file or text containing input data for prediction"
        },
    }


class TrainingParallelConfig(Config):
    """Training model parallel configuration"""
    data_parallel_size: Optional[int] = None
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    cp_comm_type: str = "all_gather"
    expert_tensor_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    sequence_parallel: bool = False

    gradient_aggregation_group: int = 1

    pipeline_parallel_config: Dict[str, Any] = {
        "pipeline_interleave": False,
        "pipeline_scheduler": "1f1b",
        "virtual_pipeline_model_parallel_size": 1,
        "seq_split_num": 1,
        "pipeline_stage_offset": 0
    }
    optimizer_parallel_config: Dict[str, Any] = {
        "enable_parallel_optimizer": False,
        "optimizer_level": "level1",
        "optimizer_weight_shard_size": -1,
        "gradient_accumulation_shard": False,
        "parallel_optimizer_threshold": 64
    }

    micro_batch_interleave_num: int = 1

    _name: str = "training_parallel_config"

    _validation_rules: Dict[str, Dict[str, Any]] = {
        "data_parallel_size": {
            "type": (int, None),
            "range": None,
            "description": ("Number of data parallel groups for distributing data "
                          "across devices")
        },
        "tensor_model_parallel_size": {
            "type": int,
            "range": None,
            "description": ("Number of tensor model parallel groups for splitting "
                          "model weights across devices")
        },
        "pipeline_model_parallel_size": {
            "type": int,
            "range": None,
            "description": ("Number of pipeline model parallel stages for splitting "
                          "model layers across devices")
        },
        "context_parallel_size": {
            "type": int,
            "range": None,
            "description": ("Number of context parallel groups for efficient "
                          "attention computation")
        },
        "cp_comm_type": {
            "type": str,
            "range": None,
            "description": ("Communication type for context parallel, "
                          "e.g., 'all_gather' for collecting activations")
        },
        "expert_model_parallel_size": {
            "type": int,
            "range": None,
            "description": ("Number of expert parallel groups for MoE models "
                          "to distribute experts")
        },
        "expert_tensor_parallel_size": {
            "type": int,
            "range": None,
            "description": ("Number of expert model parallel groups combining "
                          "tensor and expert parallelism")
        },
        "sequence_parallel": {
            "type": bool,
            "range": None,
            "description": ("Enable sequence parallelism to reduce memory usage "
                          "by splitting sequence dimension")
        },
        "pipeline_parallel_config": {
            "type": dict,
            "range": lambda x: all([
                isinstance(x.get("pipeline_interleave"), bool),
                x.get("pipeline_scheduler") in ["1f1b", "gpipe", "zero_bubble_v"],
                isinstance(x.get("virtual_pipeline_model_parallel_size"), int),
                x.get("virtual_pipeline_model_parallel_size") >= 1,
                isinstance(x.get("pipeline_stage_offset"), (int, list, str))]),
            "description": ("Pipeline parallel specific configurations including "
                          "scheduler and interleave settings")
        },
        "optimizer_parallel_config": {
            "type": dict,
            "range": lambda x: (
                isinstance(x.get("enable_parallel_optimizer"), bool)
                and x.get("optimizer_level") in ["level1", ]
                and isinstance(x.get("optimizer_weight_shard_size"), int) and not x.get("gradient_accumulation_shard")),
            "description": ("Optimizer parallel configurations for memory-efficient "
                          "optimizer state management")
        },
        "gradient_aggregation_group": {
            "type": int,
            "range": lambda x: x >= 1,
            "description": ("Group size for gradient synchronization "
                          "in data parallel training")
        },
        "micro_batch_interleave_num": {
            "type": int,
            "range": None,
            "description": ("Number of micro-batches interleaved "
                          "in pipeline parallel execution")
        },
    }


class InferParallelConfig(Config):
    """Inference model parallel configuration"""
    data_parallel_size: Optional[int] = None
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1

    _name: str = "infer_parallel_config"
    _validation_rules: Dict[str, Dict[str, Any]] = {
        "data_parallel_size": {
            "type": (int, None),
            "range": None,
            "description": ("Number of data parallel groups for distributing "
                          "inference data across devices")
        },
        "tensor_model_parallel_size": {
            "type": int,
            "range": None,
            "description": ("Number of tensor model parallel groups for splitting "
                          "model weights across devices during inference")
        },
        "pipeline_model_parallel_size": {
            "type": int,
            "range": None,
            "description": ("Number of pipeline model parallel stages for splitting "
                          "model layers across devices during inference")
        },
        "expert_model_parallel_size": {
            "type": int,
            "range": None,
            "description": ("Number of expert parallel groups for MoE models "
                          "to distribute experts during inference")
        },
    }


class RecomputeConfig(Config):
    """Recompute configuration"""
    recompute: Union[bool, List[str]] = False
    parallel_optimizer_comm_recompute: bool = True
    mp_comm_recompute: bool = True
    recompute_slice_activation: bool = False
    select_recompute: Union[bool, List[str]] = False
    select_recompute_exclude: Union[bool, List[str]] = False
    select_comm_recompute: bool = False
    select_comm_recompute_exclude: bool = False

    _name: str = "recompute_config"
    _validation_rules: Dict[str, Dict[str, Any]] = {
        "recompute": {
            "type": (bool, list),
            "range": None,
            "description": ("Enable gradient checkpointing to save memory by recomputing "
                          "activations during backward pass instead of storing them")
        },
        "select_recompute": {
            "type": (bool, list),
            "range": None,
            "description": ("Enable selective recomputation for specific layers or operations "
                          "to optimize memory usage")
        },
        "parallel_optimizer_comm_recompute": {
            "type": bool,
            "range": None,
            "description": ("Enable recomputation for communication operations in parallel "
                          "optimizer to reduce memory footprint")
        },
        "select_comm_recompute": {
            "type": bool,
            "range": None,
            "description": ("Enable selective recomputation for communication operations "
                          "to optimize memory usage")
        },
        "select_recompute_exclude": {
            "type": (bool, list),
            "range": None,
            "description": ("List of layer names or patterns to exclude "
                          "from selective recomputation")
        },
        "mp_comm_recompute": {
            "type": bool,
            "range": None,
            "description": ("Enable recomputation for model parallel communication operations "
                          "to save memory")
        },
        "recompute_slice_activation": {
            "type": bool,
            "range": [False],
            "description": ("Enable activation slicing during recomputation "
                          "to further reduce memory usage")
        },
        "select_comm_recompute_exclude": {
            "type": bool,
            "range": None,
            "description": ("Enable exclusion of specific communication operations "
                          "from selective recomputation")
        },
    }


class SwapConfig(Config):
    """Swap configuration"""
    swap: bool = False
    layer_swap: Optional[List[str]] = None
    op_swap: Optional[List[str]] = None
    default_prefetch: int = 1

    _name: str = "swap_config"
    _validation_rules: Dict[str, Dict[str, Any]] = {
        "swap": {
            "type": bool,
            "range": None,
            "description": ("Enable parameter swapping between CPU and GPU memory "
                          "to handle large models with limited GPU memory")
        },
        "layer_swap": {
            "type": (None, list),
            "range": None,
            "description": ("List of layer names or patterns to be swapped between "
                          "CPU and GPU memory during training")
        },
        "op_swap": {
            "type": (None, list),
            "range": None,
            "description": ("List of operation names or patterns to be swapped between "
                          "CPU and GPU memory during training")
        },
        "default_prefetch": {
            "type": int,
            "range": None,
            "description": ("Number of layers or operations to prefetch from CPU to GPU memory "
                          "in advance for performance optimization")
        }
    }


class TrainingConfig(Config):
    """Training configuration"""
    micro_batch_size: int = 1
    global_batch_size: int = 512
    epochs: int = 1
    training_seed: int = 1234
    dataset_seed: int = 1234

    check_for_nan_in_loss_and_grad: bool = False
    calculate_per_token_loss: bool = False
    print_separate_loss: bool = True

    scale_sense: float = 1.0
    use_clip_grad: bool = True
    max_grad_norm: float = 1.0

    gradient_accumulation_steps: int = 1
    stop_step: Optional[int] = None

    resume_training: bool = False
    data_skip_steps: Optional[int] = None
    ignore_data_skip: bool = False

    use_skip_data_by_global_norm: bool = False
    use_fast_process_recovery_by_global_norm: bool = False
    global_norm_spike_threshold: float = 1.0
    global_norm_spike_count_threshold: int = 10

    use_checkpoint_health_monitor: bool = False
    embedding_local_norm_threshold: float = 1.0

    sink_mode: bool = True
    sink_size: int = 1

    _name: str = "training_args"

    _validation_rules: Dict[str, Dict[str, Any]] = {
        "micro_batch_size": {
            "type": int,
            "range": (1, 1e32),
            "description": "Number of samples processed in one forward/backward pass during training"
        },
        "global_batch_size": {
            "type": int,
            "range": (1, 1e32),
            "description": "Total batch size across all devices/data parallel groups in distributed training"
        },
        "epochs": {
            "type": int,
            "range": (1, 1e32),
            "description": "Number of complete passes through the entire training dataset"
        },
        "training_seed": {
            "type": (None, int),
            "range": None,
            "description": "Random seed for initializing model weights and other random operations during training"
        },
        "dataset_seed": {
            "type": int,
            "range": None,
            "description": "Random seed for data shuffling and sampling to ensure reproducible data ordering"
        },
        "check_for_nan_in_loss_and_grad": {
            "type": bool,
            "range": None,
            "description": "Enable checking for NaN values in loss and gradients to detect training instability"
        },
        "calculate_per_token_loss": {
            "type": bool,
            "range": None,
            "description": "Calculate loss per token instead of per sequence for more granular loss computation"
        },
        "print_separate_loss": {
            "type": bool,
            "range": None,
            "description": "Print individual loss components separately for better training monitoring"
        },
        "use_clip_grad": {
            "type": bool,
            "range": None,
            "description": "Enable gradient clipping to prevent gradient explosion during training"
        },
        "max_grad_norm": {
            "type": float,
            "range": None,
            "description": "Maximum norm value for gradient clipping to stabilize training"
        },
        "gradient_accumulation_steps": {
            "type": int,
            "range": None,
            "description": ("Number of steps to accumulate gradients before updating model parameters, "
                          "effectively increasing batch size")
        },
        "stop_step": {
            "type": (int, None),
            "range": None,
            "description": "Maximum number of training steps after which training will be stopped"
        },
        "resume_training": {
            "type": bool,
            "range": None,
            "description": ("Whether to resume training from a checkpoint "
                          "rather than starting from scratch")
        },
        "data_skip_steps": {
            "type": (int, None),
            "range": None,
            "description": ("Number of data batches to skip at the beginning "
                          "of training when resuming")
        },
        "ignore_data_skip": {
            "type": bool,
            "range": None,
            "description": ("Ignore data skipping even when resuming training, "
                          "process all data from the beginning")
        },
        "use_skip_data_by_global_norm": {
            "type": bool,
            "range": None,
            "description": ("Enable skipping training steps based on gradient norm thresholds "
                          "for training stability")
        },
        "use_fast_process_recovery_by_global_norm": {
            "type": bool,
            "range": None,
            "description": ("Enable fast recovery mechanism when encountering "
                          "gradient norm anomalies")
        },
        "global_norm_spike_threshold": {
            "type": float,
            "range": None,
            "description": ("Gradient norm threshold above which training steps will be skipped "
                          "or recovery triggered")
        },
        "global_norm_spike_count_threshold": {
            "type": int,
            "range": None,
            "description": ("Maximum allowed gradient norm spikes before triggering "
                          "recovery mechanisms")
        },
        "use_checkpoint_health_monitor": {
            "type": bool,
            "range": None,
            "description": "Enable monitoring of checkpoint health and integrity during training"
        },
        "embedding_local_norm_threshold": {
            "type": float,
            "range": None,
            "description": "Threshold for embedding layer norm values to detect potential training issues"
        },
        "enable_expert_relocation": {
            "type": bool,
            "range": None,
            "description": "Enable dynamic relocation of expert layers in MoE models for load balancing"
        },
        "sink_mode": {
            "type": bool,
            "range": [True],
            "description": ("Enable data sinking mode to improve training performance "
                          "by reducing host-device communication")
        },
        "sink_size": {
            "type": int,
            "range": lambda x: x == 1,
            "description": "Number of iterations to sink data for performance optimization"
        },
    }


class ParallelContextConfig(Config):
    """Parallel context configuration for mindspore.set_auto_parallel_context()"""
    parallel_mode: Union[int, str] = 1
    full_batch: bool = False
    gradients_mean: bool = False
    enable_alltoall: bool = True
    search_mode: str = "sharding_propagation"

    _raise_error_for_unexpected_key: bool = False
    _name: str = "parallel_context"

    _validation_rules: Dict[str, Dict[str, Any]] = {
        "parallel_mode": {
            "type": (int, str),
            "range": [0, 1, 2],
            "description": ("Parallel execution mode for distributed training: 0-DATA_PARALLEL, "
                          "1-SEMI_AUTO_PARALLEL, 2-AUTO_PARALLEL for explicit sharding")
        },
        "full_batch": {
            "type": bool,
            "range": [False, True],
            "description": ("Whether to use full batch size across all devices "
                          "for data loading and processing")
        },
        "gradients_mean": {
            "type": bool,
            "range": None,
            "description": ("Whether to average gradients across all devices "
                          "during distributed training")
        },
        "enable_alltoall": {
            "type": bool,
            "range": None,
            "description": ("Enable AlltoAll communication primitive for efficient data exchange "
                          "in model parallel scenarios")
        },
        "search_mode": {
            "type": str,
            "range": ["recursive_programming", "sharding_propagation"],
            "description": ("Strategy search mode for automatic parallelism: 'recursive_programming' "
                          "for exhaustive search, 'sharding_propagation' for efficient propagation")
        },
    }


class InferParallelContextConfig(Config):
    """Inference parallel context configuration for mindspore.set_auto_parallel_context()"""
    parallel_mode: str = "MANUAL_PARALLEL"
    enable_alltoall: bool = False

    _raise_error_for_unexpected_key: bool = False
    _name: str = "infer_parallel_context"

    _validation_rules: Dict[str, Dict[str, Any]] = {
        "parallel_mode": {
            "type": str,
            "range": ["MANUAL_PARALLEL"],
            "description": ("Parallel execution mode for distributed inference, "
                          "only MANUAL_PARALLEL is supported for explicit sharding control")
        },
        "enable_alltoall": {
            "type": bool,
            "range": None,
            "description": ("Enable AlltoAll communication primitive for efficient data exchange "
                          "in model parallel inference scenarios")
        },
    }


class ContextConfig(Config):
    """Context configuration"""
    mode: int = 0
    device_target: str = "Ascend"
    device_id: int = 0
    max_device_memory: str = "58GB"
    mempool_block_size: str = "1GB"
    memory_optimize_level: str = "O0"
    jit_config: Dict[str, str] = {"jit_level": "O0"}
    ascend_config: Dict[str, str] = {
        "precision_mode": "must_keep_origin_dtype",
    }

    max_call_depth: int = 10000
    save_graphs: bool = False
    save_graphs_path: str = "./graph"
    enable_graph_kernel: bool = False

    affinity_cpu_list: list = None

    _raise_error_for_unexpected_key: bool = False
    _name: str = "context"

    _validation_rules: Dict[str, Dict[str, Any]] = {
        "mode": {
            "type": int,
            "range": [0, 1],
            "description": ("Execution mode for MindSpore: 0 for graph mode (optimized static graph), "
                          "1 for pynative mode (dynamic execution)")
        },
        "device_target": {
            "type": str,
            "range": ["Ascend"],
            "description": ("Target device for computation, "
                          "currently only Ascend AI processor is supported")
        },
        "device_id": {
            "type": int,
            "range": None,
            "description": ("Device ID to specify which physical device to use "
                          "when multiple devices are available")
        },
        "max_device_memory": {
            "type": str,
            "range": None,
            "description": ("Maximum memory allocation limit for the device, "
                          "specified in GB or MB (e.g., '58GB' or '59392MB')")
        },
        "mempool_block_size": {
            "type": str,
            "range": None,
            "description": ("Memory pool block size for dynamic memory allocation, "
                          "affects memory allocation efficiency")
        },
        "memory_optimize_level": {
            "type": str,
            "range": ["O0", "O1"],
            "description": ("Memory optimization level: 'O0' for basic optimization, "
                          "'O1' for aggressive memory optimization")
        },
        "jit_config": {
            "type": dict,
            "range": lambda x: x.get("jit_level") in ["O0", "O1"],
            "description": ("Just-In-Time compilation configuration, "
                          "controls compilation optimization level")
        },
        "ascend_config": {
            "type": dict,
            "range": lambda x: (
                x.get("precision_mode") in ["must_keep_origin_dtype"]
                and (x.get("parallel_speed_up_json_path") is None
                     or isinstance(x.get("parallel_speed_up_json_path"), str))),
            "description": ("Ascend-specific configuration including precision control "
                          "and parallel optimization settings")
        },
        "max_call_depth": {
            "type": int,
            "range": None,
            "description": ("Maximum recursive call depth allowed in the computation graph, "
                          "prevents stack overflow")
        },
        "affinity_cpu_list": {
            "type": (list, None),
            "range": None,
            "description": ("CPU affinity list for the computation graph, "
                          "used to bind specific CPU cores to the computation graph")
        },
        "save_graphs": {
            "type": bool,
            "range": None,
            "description": ("Whether to save graphs for debugging and analysis, "
                          "disabled by default for performance optimization")
        },
        "save_graphs_path": {
            "type": str,
            "range": None,
            "description": ("Path to save generated graphs for debugging and analysis, "
                          "defaults to './graph' directory")
        },
        "enable_graph_kernel": {
            "type": bool,
            "range": None,
            "description": ("Enable graph kernel optimization for improved performance, "
                          "disabled by default for compatibility")}
    }


class InferContextConfig(Config):
    """Inference context configuration"""
    mode: int = 0
    device_target: str = "Ascend"
    device_id: int = 0
    max_device_memory: str = "59GB"
    jit_config: Dict[str, str] = {"jit_level": "O0"}
    ascend_config: Dict[str, str] = {
        "precision_mode": "must_keep_origin_dtype",
    }

    _raise_error_for_unexpected_key: bool = False
    _name: str = "infer_context"

    _validation_rules: Dict[str, Dict[str, Any]] = {
        "mode": {
            "type": int,
            "range": [0, 1],
            "description": ("Execution mode for MindSpore inference: 0 for graph mode (optimized static graph), "
                          "1 for pynative mode (dynamic execution)")
        },
        "device_target": {
            "type": str,
            "range": ["Ascend"],
            "description": ("Target device for inference computation, "
                          "currently only Ascend AI processor is supported")
        },
        "device_id": {
            "type": int,
            "range": None,
            "description": ("Device ID to specify which physical device to use for inference "
                          "when multiple devices are available")
        },
        "max_device_memory": {
            "type": str,
            "range": None,
            "description": ("Maximum memory allocation limit for the inference device, "
                          "specified in GB or MB (e.g., '59GB' or '60416MB')")
        },
        "jit_config": {
            "type": dict,
            "range": lambda x: x.get("jit_level") in ["O0"],
            "description": ("Just-In-Time compilation configuration for inference, "
                          "controls compilation optimization level")
        },
        "ascend_config": {
            "type": dict,
            "range": lambda x: x.get("precision_mode") in ["must_keep_origin_dtype"],
            "description": ("Ascend-specific inference configuration "
                          "including precision control settings")
        },
    }


class MegatronDataLoaderConfig(Config):
    """Megatron data loader configuration"""
    type: str = "BlendedMegatronDatasetDataLoader"
    datasets_type: str = "GPTDataset"
    config: Dict[str, Any] = {
        "seed": 1234,
        "split": "1, 0, 0",
        "eod_mask_loss": True,
        "reset_position_ids": True,
        "create_attention_mask": True,
        "reset_attention_mask": True,
        "create_compressed_eod_mask": False,
        "compressed_eod_mask_length": 128
    }

    _raise_error_for_unexpected_key: bool = False
    _name: str = "megatron_dataloader"
    _required_keys: Dict[str, List[str]] = {
        "megatron_dataloader": ["type", "sizes", "config"],
        "config": []
    }

    _validation_rules: Dict[str, Dict[str, Any]] = {
        "type": {
            "type": str,
            "range": ["BlendedMegatronDatasetDataLoader"],
            "description": "Data loader type, must be 'BlendedMegatronDatasetDataLoader' for Megatron data processing"
        },
        "datasets_type": {
            "type": str,
            "range": ["GPTDataset"],
            "description": "Type of dataset to load, e.g., 'GPTDataset' for GPT-style language modeling datasets"
        },
        "config": {
            "type": dict,
            "range": None,
            "description": "Configuration dictionary for the data loader with detailed dataset processing parameters"
        }
    }


class HFDataLoader(Config):
    """HuggingFace data loader configuration"""
    type: str = "HFDataLoader"
    shuffle: bool = False
    create_attention_mask: bool = True
    create_compressed_eod_mask: bool = False
    compressed_eod_mask_length: int = 128
    use_broadcast_data: bool = True
    split: str = "train"
    _raise_error_for_unexpected_key: bool = False
    _name: str = "hf_dataloader"
    _required_keys: Dict[str, List[str]] = {
        "hf_dataloader": ["type", "load_func", "path", "data_files", "handler"]
    }

    _validation_rules: Dict[str, Dict[str, Any]] = {
        "type": {
            "type": str,
            "range": ["HFDataLoader"],
            "description": "Data loader type, must be 'HFDataLoader' for HuggingFace dataset processing"
        },
        "shuffle": {
            "type": bool,
            "range": None,
            "description": "Whether to shuffle the dataset during training"
        },
        "create_attention_mask": {
            "type": bool,
            "range": None,
            "description": "Whether to create attention masks for the input sequences"
        },
        "create_compressed_eod_mask": {
            "type": bool,
            "range": None,
            "description": "Whether to create compressed end-of-document masks for efficient processing"
        },
        "compressed_eod_mask_length": {
            "type": int,
            "range": None,
            "description": "Length of the compressed end-of-document mask"
        },
        "use_broadcast_data": {
            "type": bool,
            "range": None,
            "description": "Whether to use broadcast data distribution in distributed training"
        },
        "split": {
            "type": str,
            "range": ["train"],
            "description": "Dataset split to use, e.g., 'train', 'validation', or 'test'"
        }
    }


class TrainingDatasetConfig(Config):
    """Training dataset configuration"""
    num_parallel_workers: int = 8
    python_multiprocessing: bool = False
    drop_remainder: bool = True
    numa_enable: bool = False
    prefetch_size: int = 1
    input_columns: Optional[List[str]] = None
    construct_args_key: Optional[List[str]] = None

    _raise_error_for_unexpected_key: bool = False
    _name: str = "train_dataset"
    _required_keys: Dict[str, List[str]] = {
        "train_dataset": ["data_loader"]
    }
    _exclude_recursive_dict_key: List[str] = ["data_loader"]

    _validation_rules: Dict[str, Dict[str, Any]] = {
        "num_parallel_workers": {
            "type": int,
            "range": None,
            "description": ("Number of parallel workers for data loading "
                          "to accelerate dataset processing")
        },
        "python_multiprocessing": {
            "type": bool,
            "range": None,
            "description": ("Whether to use Python multiprocessing for data loading "
                          "to improve performance")
        },
        "drop_remainder": {
            "type": bool,
            "range": None,
            "description": ("Whether to drop the last incomplete batch when dataset size "
                          "is not divisible by batch size")
        },
        "numa_enable": {
            "type": bool,
            "range": None,
            "description": ("Whether to enable NUMA (Non-Uniform Memory Access) optimization "
                          "for data loading")
        },
        "prefetch_size": {
            "type": int,
            "range": None,
            "description": ("Number of batches to prefetch for pipeline parallelism "
                          "between data loading and model training")
        },
        "input_columns": {
            "type": (list, None),
            "range": None,
            "description": ("List of column names from the dataset to be used as input "
                          "for the model training process")
        },
        "construct_args_key": {
            "type": (list, None),
            "range": None,
            "description": ("List of argument keys used to construct the input arguments "
                          "for model's forward pass during training")
        }
    }


class TrainerConfig(Config):
    """Trainer configuration"""
    _raise_error_for_unexpected_key: bool = False

    _name: str = "trainer"
    _required_keys: Dict[str, List[str]] = {
        "trainer": ["type"]
    }


class ModelConfig(Config):
    """Model configuration"""
    _raise_error_for_unexpected_key: bool = False
    _name: str = "model_config"


class OptimizerConfig(SpecConfig):
    """Optimizer configuration"""
    _name: str = "optimizer"
    _required_keys: Dict[str, List[str]] = {
        "optimizer": ["type"]
    }

    type: str = "AdamW"
    betas: List[float] = [0.9, 0.999]
    learning_rate: float = 5.e-5
    eps: float = 1.e-8
    weight_decay: float = 0.0

    _raise_error_for_unexpected_key: bool = False


class LrScheduleConfig(SpecConfig):
    """Learning rate schedule configuration"""
    _name: str = "lr_schedule"
    _required_keys: Dict[str, List[str]] = {
        "lr_schedule": ["type"]
    }

    type: str = "CosineWithWarmUpLR"
    learning_rate: float = 5.e-5
    lr_end: float = 0.
    warmup_ratio: float = 0.
    total_steps: int = -1

    _raise_error_for_unexpected_key: bool = False


class CallbackConfig(ListConfig):
    """Callback configuration"""
    _name: str = "callbacks"
    _required_keys: Dict[str, List[str]] = {
        "callbacks": ["type"]
    }


class MonitorConfig(Config):
    """Monitor configuration"""
    dump_path: str = "./dump"
    local_loss: Union[str, List[str]] = None
    device_local_norm: Union[str, List[str]] = None
    device_local_loss: Union[str, List[str]] = None

    local_norm: Optional[Any] = None
    optimizer_params_state: Optional[Any] = None
    net_weight_params_state: Optional[Any] = None
    target_parameters: Optional[List[str]] = [".*"]
    target_parameters_invert: bool = False

    embedding_local_norm: bool = False

    step_interval: int = 1

    _name: str = "monitor_config"
    _raise_error_for_unexpected_key: bool = True

    _validation_rules: Dict[str, Dict[str, Any]] = {
        "dump_path": {
            "type": str,
            "range": None,
            "description": "Path to dump monitoring data and logs for debugging and analysis"
        },
        "step_interval": {
            "type": int,
            "range": None,
            "description": "Interval (in steps) at which to collect and dump monitoring information"
        },
        "local_loss": {
            "type": (None, str, list),
            "range": None,
            "description": "Local loss monitoring configuration for tracking training loss on each device"
        },
        "device_local_norm": {
            "type": (None, str, list),
            "range": None,
            "description": "Device local norm monitoring configuration for tracking gradient norms on each device"
        },
        "device_local_loss": {
            "type": (None, str, list),
            "range": None,
            "description": "Device local loss monitoring configuration for detailed per-device loss tracking"
        },
        "local_norm": {
            "type": (None, Any),
            "range": None,
            "description": "Local norm monitoring configuration for tracking parameter norms"
        },
        "optimizer_params_state": {
            "type": (None, Any),
            "range": None,
            "description": "Optimizer parameters state monitoring configuration for tracking optimizer internal states"
        },
        "net_weight_params_state": {
            "type": (None, Any),
            "range": None,
            "description": "Network weight parameters state monitoring configuration for tracking model weights"
        },
        "target_parameters": {
            "type": list,
            "range": None,
            "description": "List of parameter name patterns to monitor, supports regular expressions"
        },
        "target_parameters_invert": {
            "type": bool,
            "range": None,
            "description": ("If True, invert the target parameter selection, monitoring all parameters "
                          "except those matching the patterns")
        },
        "embedding_local_norm": {
            "type": bool,
            "range": None,
            "description": ("Enable monitoring of embedding layer local norms "
                          "for tracking embedding parameter stability")
        }
    }


class TensorBoardConfig(Config):
    """TensorBoard configuration"""
    tensorboard_on: bool = False
    tensorboard_dir: str = './tensorboard'
    tensorboard_queue_size: int = 10
    log_loss_scale_to_tensorboard: bool = False
    log_timers_to_tensorboard: bool = False
    log_expert_load_to_tensorboard: bool = False

    _name: str = "tensorboard"
    _raise_error_for_unexpected_key: bool = True

    _validation_rules: Dict[str, Dict[str, Any]] = {
        "tensorboard_on": {
            "type": bool,
            "range": None,
            "description": ("Whether to enable TensorBoard logging "
                          "for training visualization and monitoring")
        },
        "tensorboard_dir": {
            "type": str,
            "range": None,
            "description": "Directory path to store TensorBoard log files for visualization"
        },
        "tensorboard_queue_size": {
            "type": int,
            "range": None,
            "description": ("Size of the queue for buffering TensorBoard events "
                          "before writing to disk")
        },
        "log_loss_scale_to_tensorboard": {
            "type": bool,
            "range": None,
            "description": ("Whether to log loss scaling information to TensorBoard "
                          "for mixed precision training monitoring")
        },
        "log_timers_to_tensorboard": {
            "type": bool,
            "range": None,
            "description": ("Whether to log timing information to TensorBoard "
                          "for performance analysis")
        },
        "log_expert_load_to_tensorboard": {
            "type": bool,
            "range": None,
            "description": ("Whether to log expert load balancing information to TensorBoard "
                          "for MoE model monitoring")
        }
    }


class ProfileConfig(Config):
    """Profile configuration"""
    profile_on: bool = False
    profile_output: Optional[str] = None
    profiler_level: int = 1
    profile_start_step: int = 1
    profile_stop_step: int = 10
    init_start_profile: bool = False
    profile_rank_ids: Optional[List[int]] = None  # List[int]
    profile_pipeline: bool = False
    profile_communication: bool = False
    profile_memory: bool = True
    with_stack: bool = False
    data_simplification: bool = False
    mstx: bool = False
    use_llm_token_profile: bool = False
    llm_token_profile_config: Optional[dict] = None
    _name: str = "profile"
    _raise_error_for_unexpected_key: bool = True
    _exclude_recursive_dict_key: List[str] = ["llm_token_profile_config"]

    _validation_rules: Dict[str, Dict[str, Any]] = {
        "profile_on": {
            "type": bool,
            "range": None,
            "description": "Enable performance profiling to analyze model execution performance and bottlenecks"
        },
        "profile_output": {
            "type": (str, None),
            "range": None,
            "description": "Directory path to save profiling results, if None uses default path"
        },
        "profiler_level": {
            "type": int,
            "range": [1],
            "description": "Profiling level, currently only level 1 is supported for basic profiling"
        },
        "profile_start_step": {
            "type": int,
            "range": None,
            "description": "Training step at which to start profiling data collection"
        },
        "profile_stop_step": {
            "type": int,
            "range": None,
            "description": "Training step at which to stop profiling data collection"
        },
        "init_start_profile": {
            "type": bool,
            "range": None,
            "description": "Whether to start profiling immediately from the initialization phase"
        },
        "profile_rank_ids": {
            "type": (list, None),
            "range": None,
            "description": "List of device rank IDs to profile, if None profiles all ranks"
        },
        "profile_pipeline": {
            "type": bool,
            "range": None,
            "description": "Enable pipeline parallel profiling to analyze pipeline execution efficiency"
        },
        "profile_communication": {
            "type": bool,
            "range": None,
            "description": "Enable communication operation profiling to analyze network performance"
        },
        "profile_memory": {
            "type": bool,
            "range": None,
            "description": "Enable memory usage profiling to analyze memory consumption patterns"
        },
        "with_stack": {
            "type": bool,
            "range": None,
            "description": "Include stack trace information in profiling results for detailed analysis"
        },
        "data_simplification": {
            "type": bool,
            "range": None,
            "description": "Simplify profiling data to reduce output file size"
        },
        "mstx": {
            "type": bool,
            "range": None,
            "description": "Enable MSTX (MindSpore Trace eXtension) profiling for enhanced analysis"
        },
        "use_llm_token_profile": {
            "type": bool,
            "range": None,
            "description": "Enable LLM token-level profiling to analyze token distribution and processing"
        },
        "llm_token_profile_config": {
            "type": (dict, None),
            "range": None,
            "description": "Configuration for LLM token profiling, specifies detailed token analysis settings"
        }
    }


class CheckpointConfig(Config):
    """Checkpoint configuration"""
    # 权重加载
    load_checkpoint: str = ''
    load_ckpt_format: str = 'safetensors'
    balanced_load: bool = False

    # 权重保存
    prefix: str = "llm_model"
    save_checkpoint_seconds: int = 0
    save_checkpoint_steps: int = 1
    keep_checkpoint_max: int = 1
    keep_checkpoint_per_n_minutes: int = 0
    integrated_save: bool = False
    save_network_params: bool = False
    save_trainable_params: bool = False
    async_save: bool = False

    # 去冗余保存或加载开关
    remove_redundancy: bool = True

    _name: str = "checkpoint_config"
    _required_keys: Dict[str, List[str]] = {
        "checkpoint_config": ["save_checkpoint_steps"]
    }

    _validation_rules: Dict[str, Dict[str, Any]] = {
        "load_checkpoint": {
            "type": (str, None),
            "range": None,
            "description": "Path to the checkpoint file or directory for loading model weights"
        },
        "load_ckpt_format": {
            "type": str,
            "range": ["safetensors"],
            "description": "Format of the checkpoint file to load, e.g., 'safetensors' for secure tensor storage"
        },
        "balanced_load": {
            "type": bool,
            "range": [False],
            "description": "Enable balanced loading of checkpoints across devices in distributed training"
        },
        "prefix": {
            "type": str,
            "range": None,
            "description": "Prefix for checkpoint file names to identify model checkpoints"
        },
        "save_checkpoint_seconds": {
            "type": int,
            "range": None,
            "description": "Interval in seconds to save checkpoints periodically during training"
        },
        "save_checkpoint_steps": {
            "type": int,
            "range": None,
            "description": "Interval in training steps to save checkpoints periodically during training"
        },
        "keep_checkpoint_max": {
            "type": int,
            "range": None,
            "description": "Maximum number of checkpoint files to keep, older checkpoints will be deleted"
        },
        "keep_checkpoint_per_n_minutes": {
            "type": int,
            "range": None,
            "description": "Keep one checkpoint file per N minutes, 0 means no time-based checkpoint retention"
        },
        "integrated_save": {
            "type": bool,
            "range": [False],
            "description": "Enable integrated saving of all model components into a single checkpoint file"
        },
        "save_network_params": {
            "type": bool,
            "range": None,
            "description": "Save all network parameters including non-trainable parameters to checkpoint"
        },
        "save_trainable_params": {
            "type": bool,
            "range": None,
            "description": "Save only trainable parameters to checkpoint for smaller file size"
        },
        "async_save": {
            "type": bool,
            "range": [False],
            "description": "Enable asynchronous checkpoint saving to avoid blocking training process"
        },
        "remove_redundancy": {
            "type": bool,
            "range": None,
            "description": "Remove redundant data during checkpoint saving or loading to reduce file size"
        }
    }


CONFIG_NAME_TO_CLASS: Dict[str, type] = {
    "training_general_config": TrainingGeneralConfig,
    "infer_general_config": InferGeneralConfig,
    "training_parallel_config": TrainingParallelConfig,
    "infer_parallel_config": InferParallelConfig,
    "recompute_config": RecomputeConfig,
    "swap_config": SwapConfig,
    "training_args": TrainingConfig,
    "parallel": ParallelContextConfig,
    "infer_parallel_context": InferParallelContextConfig,
    "context": ContextConfig,
    "infer_context": InferContextConfig,
    "train_dataset": TrainingDatasetConfig,
    "megatron_dataloader": MegatronDataLoaderConfig,
    "hf_dataloader": HFDataLoader,
    "trainer": TrainerConfig,
    "model_config": ModelConfig,
    "optimizer": OptimizerConfig,
    "lr_schedule": LrScheduleConfig,
    "callbacks": CallbackConfig,
    "monitor_config": MonitorConfig,
    "profile": ProfileConfig,
    "tensorboard": TensorBoardConfig,
    "checkpoint_config": CheckpointConfig
}


class ConfigTemplate:
    """
    A template handler for managing and applying configurations.

    This class organizes and applies configuration templates based on different run modes
    such as 'train', 'predict', and 'finetune'. It categorizes configurations into
    general, training, evaluation, and prediction sections to structure and streamline the
    application of various configurations for workflows.

    Attributes:
        train_configs (list): A list of configuration sections specific to training.
        predict_configs (list): A list of configuration sections for prediction tasks.
        _run_modes (list): Supported modes of operation ('train', 'predict', 'finetune').
    """
    train_configs: List[str] = [
    "training_general_config",
    "distribute_parallel_config",
    "recompute_config",
    "swap_config",
    "training_args",
    "parallel",
    "context",
    "train_dataset",
    "trainer",
    "model_config",
    "optimizer",
    "lr_schedule",
    "callbacks",
    "monitor_config",
    "profile",
    "tensorboard",
    "checkpoint_config"]

    predict_configs: List[str] = [
    "infer_general_config",
    "distribute_parallel_config",
    "parallel",
    "context",
    "trainer",
    "model_config"]

    _run_modes: List[str] = ['train', 'predict', 'finetune']

    @classmethod
    def apply_template(cls, config: Dict[str, Any]) -> None:
        """
        Apply the appropriate configuration template based on the run mode.

        Args:
            config (dict): The configuration dict containing the run mode
                and other relevant settings.

        Returns:
            None
        """
        run_mode = config.get('run_mode')
        if run_mode in ['train', 'finetune']:
            template = cls._train_template()
        elif run_mode == "predict":
            template = cls._predict_template()
        else:
            raise ValueError(f"The specified run_mode '{run_mode}' is invalid. Expected one of {cls._run_modes}.")
        cls._apply_template(config, template, run_mode)


    @classmethod
    def _apply_template(cls, config: Dict[str, Any], template: List[str], run_mode: str) -> None:
        """
        Apply a specific template to the configuration.

        Args:
            config (dict): The original configuration dict.
            template (list): A list of configuration sections to be applied.
            run_mode (str): The current run mode.

        Returns:
            None
        """
        general_config = cls._aggregate_general_config(config, template, run_mode)

        new_config = {}
        for sub_config in template:
            if sub_config == "distribute_parallel_config":
                origin_sub_config = config.pop(sub_config, None)
                cls.update_distributed_parallel_config(sub_config, new_config, origin_sub_config, run_mode)
                continue

            if sub_config == "context":
                origin_sub_config = config.pop(sub_config, None)
                cls.update_context_config(sub_config, new_config, origin_sub_config, run_mode)
                continue

            if sub_config == "parallel":
                origin_sub_config = config.pop(sub_config, None)
                cls.update_parallel_context_config(sub_config, new_config, origin_sub_config, run_mode)
                continue

            if sub_config == "train_dataset":
                origin_sub_config = config.pop(sub_config, None)
                class_ = CONFIG_NAME_TO_CLASS[sub_config]
                new_config[sub_config] = class_.apply(sub_config, origin_sub_config)
                class_.validate_config(new_config[sub_config])
                cls.update_train_dataset_config(sub_config, new_config, origin_sub_config, run_mode)
                continue

            class_ = CONFIG_NAME_TO_CLASS[sub_config]
            new_config[sub_config] = class_.apply(sub_config, config.pop(sub_config, None))
            class_.validate_config(new_config[sub_config])

        unused_config = list(config.keys())
        if unused_config:
            logger.warning(f"Some configs in yaml are useless for {run_mode}: {unused_config}")
        config.update(new_config)
        config.update(general_config)

    @staticmethod
    def update_distributed_parallel_config(sub_config: str, new_config: Dict[str, Any],
                                           origin_config: Union[Dict[str, Any], None],
                                           run_mode: str) -> None:
        """
        Update distributed parallel configuration based on run mode.

        Args:
            sub_config (str): The configuration section name.
            new_config (dict): The new configuration dictionary to update.
            origin_config (dict): The original configuration.
            run_mode (str): The current run mode.

        Returns:
            None
        """
        if run_mode in ['train', 'finetune']:
            class_ = CONFIG_NAME_TO_CLASS["training_parallel_config"]
        elif run_mode == "predict":
            class_ = CONFIG_NAME_TO_CLASS["infer_parallel_config"]
        else:
            raise ValueError()
        new_config[sub_config] = class_.apply(sub_config, origin_config)
        class_.validate_config(new_config[sub_config])

    @staticmethod
    def update_parallel_context_config(sub_config: str, new_config: Dict[str, Any],
                                       origin_config: Union[Dict[str, Any], None],
                                       run_mode: str) -> None:
        """
        Update parallel context configuration based on run mode.

        Args:
            sub_config (str): The configuration section name.
            new_config (dict): The new configuration dictionary to update.
            origin_config (dict): The original configuration.
            run_mode (str): The current run mode.

        Returns:
            None
        """
        if run_mode in ['train', 'finetune']:
            class_ = CONFIG_NAME_TO_CLASS["parallel"]
        elif run_mode == "predict":
            class_ = CONFIG_NAME_TO_CLASS["infer_parallel_context"]
        else:
            raise ValueError()
        new_config[sub_config] = class_.apply(sub_config, origin_config)
        class_.validate_config(new_config[sub_config])

    @staticmethod
    def update_context_config(sub_config: str, new_config: Dict[str, Any],
                              origin_config: Union[Dict[str, Any], None], run_mode: str) -> None:
        """
        Update context configuration based on run mode.

        Args:
            sub_config (str): The configuration section name.
            new_config (dict): The new configuration dictionary to update.
            origin_config (dict): The original configuration.
            run_mode (str): The current run mode.

        Returns:
            None
        """
        if run_mode in ['train', 'finetune']:
            class_ = CONFIG_NAME_TO_CLASS["context"]
        elif run_mode == "predict":
            class_ = CONFIG_NAME_TO_CLASS["infer_context"]
        else:
            raise ValueError()
        new_config[sub_config] = class_.apply(sub_config, origin_config)
        class_.validate_config(new_config[sub_config])

    @staticmethod
    def update_train_dataset_config(sub_config: str, new_config: Dict[str, Any],
                                    origin_config: Union[Dict[str, Any], None],
                                    run_mode: str) -> None:
        """
        Update training dataset configuration based on run mode.

        Args:
            sub_config (str): The configuration section name.
            new_config (dict): The new configuration dictionary to update.
            origin_config (dict): The original configuration.
            run_mode (str): The current run mode.

        Returns:
            None
        """
        if run_mode in ['train', 'finetune']:
            if origin_config and "data_loader" in origin_config:
                data_loader_type = origin_config["data_loader"].get("type")
                if data_loader_type == "BlendedMegatronDatasetDataLoader":
                    class_ = CONFIG_NAME_TO_CLASS["megatron_dataloader"]
                    new_config[sub_config]["data_loader"] = class_.apply(
                        sub_config, origin_config.pop("data_loader", None))
                    class_.validate_config(new_config[sub_config]["data_loader"])
                elif data_loader_type == "HFDataLoader":
                    class_ = CONFIG_NAME_TO_CLASS["hf_dataloader"]
                    new_config[sub_config]["data_loader"] = class_.apply(
                        sub_config, origin_config.pop("data_loader", None))
                    class_.validate_config(new_config[sub_config]["data_loader"])

    @classmethod
    def _train_template(cls) -> List[str]:
        """
        Get the training template configuration list.

        Returns:
            list: A list of configuration section names for training.
        """
        template = []
        template.extend(cls.train_configs)
        return template

    @classmethod
    def _predict_template(cls) -> List[str]:
        """
        Get the prediction template configuration list.

        Returns:
            list: A list of configuration section names for prediction.
        """
        template = []
        template.extend(cls.predict_configs)
        return template

    @staticmethod
    def _aggregate_general_config(config: Dict[str, Any], template: List[str], run_mode: str) -> Dict[str, Any]:
        """
        Aggregate all general configuration keys into a single 'general_config' section.

        Args:
            config (dict): The original configuration dict.
            template (list): The template configuration list.
            run_mode (str): The current run mode.

        Returns:
            dict: The configuration dict with aggregated general settings.
        """
        general_config = {}
        if run_mode in ["finetune", "train"]:
            general_config_class = CONFIG_NAME_TO_CLASS["training_general_config"]
            template.remove("training_general_config")
        elif run_mode in ["predict"]:
            general_config_class = CONFIG_NAME_TO_CLASS["infer_general_config"]
            template.remove("infer_general_config")
        else:
            raise ValueError(f"The specified run_mode '{run_mode}' is invalid.")
        for key in general_config_class.keys():
            if key not in config.keys():
                general_config[key] = general_config_class.default_value().get(key)
            else:
                general_config[key] = config.pop(key)
        general_config_class.validate_config(general_config)
        return general_config
