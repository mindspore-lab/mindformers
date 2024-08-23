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
"""optimizer registration and factory method"""

from mindspore.nn import Adam, SGD
from mindformers.core.optim import Came, AdamW
from mindformers.experimental.distri_cores.register import ModuleType, ModuleRegistry
from mindformers.experimental.distri_cores.optimizer.lr_scheduler.lr_scheduler import get_learning_rate_scheduler

ModuleRegistry.register(AdamW, ModuleType.OPTIMIZER)
ModuleRegistry.register(Adam, ModuleType.OPTIMIZER)
ModuleRegistry.register(SGD, ModuleType.OPTIMIZER)
ModuleRegistry.register(Came, ModuleType.OPTIMIZER)


def get_optimizer(optimizer_config, params=None, network=None, return_instance: bool = True, **kwargs):
    """
    Get an optimizer instance or class based on the provided optimizer configuration.

    Args:
        optimizer_config (OptimizerConfig): The configuration object for the optimizer.
        params (list or dict): The parameters to optimize.
        network (nn.Cell): The network model.
        return_instance (bool): Whether to return an instance of the optimizer or just the optimizer class.
        **kwargs: Additional keyword arguments to be passed to the optimizer class.

    Returns:
        Optimizer or type: An instance of the optimizer class if `return_instance` is True,
        otherwise the optimizer class itself.

    Raises:
        ValueError: If `params` is None and `return_instance` is True.
        NotImplementedError: If `weight_decay_kwargs` is not supported yet.

    """
    if optimizer_config.parallel_config.zero_level is not None:
        optimizer_type = optimizer_config.optimizer_type + "ZeRO"
    else:
        optimizer_type = optimizer_config.optimizer_type
    optimizer_cls = ModuleRegistry.get_item(module_type=ModuleType.OPTIMIZER, item_name=optimizer_type)

    if not return_instance:
        return optimizer_cls

    if params is None:
        raise ValueError("params must be provided when return_instance is True.")

    if optimizer_config.weight_decay_kwargs is not None:
        raise NotImplementedError("weight_decay_kwargs is not supported yet.")
        # weight_decay = get_weight_decay(optimizer_config.optimizer)

    weight_decay = optimizer_config.weight_decay

    if optimizer_config.learning_rate_scheduler_kwargs is not None:
        learning_rate = get_learning_rate_scheduler(optimizer_config)
    else:
        learning_rate = optimizer_config.learning_rate

    if isinstance(params[0], dict):
        using_group_lr = any("lr" in param for param in params)
        for param in params:
            if "order_params" in param:
                continue
            if "lr" not in param and using_group_lr:
                param["lr"] = learning_rate
            if "weight_decay" not in param:
                param["weight_decay"] = weight_decay

    optimizer_kwargs = optimizer_config.get_needed_params_for_class(optimizer_cls)
    optimizer_kwargs["learning_rate"] = learning_rate
    optimizer_kwargs["weight_decay"] = weight_decay
    optimizer_kwargs["params"] = params
    if optimizer_config.parallel_config.zero_level is not None:
        if network is None:
            raise ValueError("Network must be provided when get ZeRO optimizer instance.")
        optimizer_kwargs["zero_level"] = optimizer_config.parallel_config.zero_level
        optimizer_kwargs["network"] = network
        if optimizer_config.zero_config is not None:
            optimizer_kwargs.update(optimizer_config.zero_config)
    optimizer_kwargs.update(kwargs)
    return_item = optimizer_cls(**optimizer_kwargs)

    return return_item
