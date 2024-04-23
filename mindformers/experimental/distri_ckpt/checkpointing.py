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

"""Model and parameters serialization."""
import os
import mindspore as ms
from mindspore import nn, Parameter
from mindspore import _checkparam as validator
from mindspore.communication import get_rank, get_group_size, create_group, GlobalComm
from mindspore.train.serialization import _convert_save_obj_to_param_list
from mindspore.ops.operations import Broadcast, Zeros, AllGather, ReduceScatter
from mindformers.tools.logger import logger

# Distribution configurations.
_ckpt_rank = 0
_optim_shard_size = 1
_optim_comm_group = None
_rank_size = 1
_weight_name = "mindspore_model.ckpt"


def _is_optim_states(param: Parameter):
    """Check whether the parameter is optimizer states."""
    for name in ['adam_m', 'adam_v']:
        if name in param.name:
            return True
    return False


def get_checkpoint_name(ckpt_path):
    """Get checkpoint file name of model and optimizer."""
    validator.check_value_type("ckpt_path", ckpt_path, [str])
    ckpt_path = os.path.abspath(ckpt_path)
    ckpt_path = os.path.normpath(ckpt_path)
    # ensure that file path is exist
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_name = os.path.join(ckpt_path, _weight_name)
    return ckpt_name


def save_checkpoint(model, optimizer=None, ckpt_path="./", **kwargs):
    """
    Save checkpoint of distributed network to a specified file in the process of specified rank.

    Args:
        model (Cell): The network to be saved.
        ckpt_path (str): Checkpoint file path. Default: ``"./"``.
        optimizer (Cell): The optimizer to be saved. Default: ``None``.
        kwargs (dict): Configuration options dictionary.

    Raises:
        TypeError: If the type of parameter `model` is not nn.Cell.
        TypeError: If the type of parameter `optimizer` is not nn.Cell.
        TypeError: If the type of parameter `ckpt_path` is not str.
    """
    validator.check_value_type("model", model, [nn.Cell], "save_checkpoint")
    validator.check_value_type("optimizer", optimizer, [nn.Cell, type(None)], "save_checkpoint")
    ckpt_name = get_checkpoint_name(ckpt_path)
    for key, _ in kwargs.items():
        logger.warning(f"The parameter {key} is not used in save_checkpoint.")
    if optimizer:
        param_list = _convert_save_obj_to_param_list(optimizer, True, None, None)
    else:
        param_list = _convert_save_obj_to_param_list(model, True, None, None)
    # optimizer parallel is not enabled
    if not optimizer or not hasattr(optimizer, "use_parallel") or not optimizer.use_parallel:
        ms.save_checkpoint(param_list, ckpt_name)
        return
    # update distributed config
    global _optim_comm_group
    _optim_comm_group = optimizer.opt_parallel_group
    # optimizer parallel is enabled
    for param_dict in param_list:
        # gather states to process of _ckpt_rank
        if _is_optim_states(param_dict.get('data')):
            param_dict['data'] = AllGather(group=_optim_comm_group)(param_dict['data'])
    # save checkpoint in process of _ckpt_rank
    if get_rank() == _ckpt_rank:
        ms.save_checkpoint(param_list, ckpt_name)


def load_checkpoint(ckpt_path, model, optimizer=None, **kwargs):
    """
    Load checkpoint info from a specified file in process of rank 0.

    Args:
        ckpt_path (str): Checkpoint file path.
        model (Cell): The network where the parameters will be loaded.
        optimizer (Cell): The optimizer where the parameters will be loaded.

    Raises:
        TypeError: If the type of parameter `ckpt_path` is not str.
        TypeError: If the type of parameter `model` is not nn.Cell.
        TypeError: If the type of parameter `optimizer` is not nn.Cell. Default: ``None``.
    """
    validator.check_value_type("model", model, [nn.Cell], "load_checkpoint")
    validator.check_value_type("optimizer", optimizer, [nn.Cell, type(None)], "load_checkpoint")
    ckpt_name = get_checkpoint_name(ckpt_path)
    for key, _ in kwargs.items():
        logger.warning(f"The parameter {key} is not used in load_checkpoint.")
    # optimizer parallel is not enabled
    if not optimizer or not hasattr(optimizer, "use_parallel") or not optimizer.use_parallel:
        ms.load_checkpoint(ckpt_name, model)
        if optimizer:
            ms.load_checkpoint(ckpt_name, optimizer)
        return
    # optimizer parallel is enabled, update distributed config
    global _optim_shard_size, _optim_comm_group, _rank_size
    _rank_size = get_group_size(GlobalComm.WORLD_COMM_GROUP)
    _optim_comm_group = optimizer.opt_parallel_group
    _optim_shard_size = get_group_size(_optim_comm_group)
    # load checkpoint in process of _ckpt_rank
    if get_rank() == _ckpt_rank:
        param_dict = ms.load_checkpoint(ckpt_name)
    else:
        param_dict = {}
    # update parameters
    param_loaded = _load_param_into_net(model, param_dict)
    if optimizer:
        _load_param_into_net(optimizer, param_dict, param_loaded)


def _load_param_into_net(net: nn.Cell, parameter_dict: dict, param_loaded: list = None):
    """Load parameters into network."""
    if param_loaded is None:
        param_loaded = []
    # initialize parameters
    net.init_parameters_data()
    # broadcast/scatter parameters
    for _, param in net.parameters_and_names():
        if param.name in param_loaded:
            continue
        new_param = parameter_dict[param.name] if param.name in parameter_dict else param
        if _is_optim_states(param):
            # for optimizer states in other process, a full shape tensor is used for communication operations
            if get_rank() != _ckpt_rank:
                param_shape = list(new_param.shape)
                param_shape[0] *= _optim_shard_size
                new_param = Zeros()(tuple(param_shape), new_param.dtype)
            # broadcast full parameters to first process of each group
            if _optim_shard_size < _rank_size and get_rank() % _optim_shard_size == 0:
                rank_ids = [i for i in range(_rank_size) if i % _optim_shard_size == 0]
                create_group("optim_states_group", rank_ids)
                new_param = Broadcast(root_rank=_ckpt_rank, group="optim_states_group")((new_param,))[0]
            new_tensor = ReduceScatter(group=_optim_comm_group)(new_param)
        else:
            new_tensor = Broadcast(root_rank=_ckpt_rank, group=GlobalComm.WORLD_COMM_GROUP)((new_param,))[0]
        param.assign_value(new_tensor)
        param_loaded.append(param.name)
    return param_loaded
