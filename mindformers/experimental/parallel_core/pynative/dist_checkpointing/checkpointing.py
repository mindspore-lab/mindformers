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
import glob
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import _checkparam as validator
from mindspore.ops import operations as P
try:
    from mindspore import default_generator, set_rng_state
except ImportError:
    from mindspore.nn.generator import default_generator, set_rng_state
from mindspore.communication import get_rank
from mindformers.tools.logger import logger
from mindformers.experimental.parallel_core.pynative.utils import generate_state_dict, save_strategy_file
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_pipeline_model_parallel_world_size,
    get_expert_model_parallel_world_size
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel.random import (
    get_rng_tracer,
    CANDIDATE_MODES
)

__all__ = ["get_checkpoint_name",
           "save_rng_state",
           "load_rng_state",
           'save_pre_process',
           'load_post_process',
           "save_checkpoint",
           "load_checkpoint"]

# Distribution configurations.
_strategy_dir = "strategy"
_format = "safetensors"

# pylint: disable=W0622
def get_checkpoint_name(ckpt_path, format=_format, get_name_from_file=False):
    """
    Get checkpoint file name of model and optimizer.
    The layout of the ckpt_path will be like:
        ckpt_path/
        ├── rank_0
        │   ├── network0.ckpt
        ├── rank_1
        │   ├── network1.ckpt
        └── strategy
            ├── strategy0.ckpt
            └── strategy1.ckpt
    The strategy file will be saved in a standalone dir for the possible subsequent merging.
    The checkpoint file will be separated in different dir for the possible subsequent transformation.
    """
    validator.check_value_type("ckpt_path", ckpt_path, [str])
    rank = get_rank()
    # ensure ckpt path exist
    ckpt_path = os.path.normpath(os.path.abspath(ckpt_path))
    ckpt_local_path = os.path.join(ckpt_path, f"rank_{rank}")
    os.makedirs(ckpt_local_path, exist_ok=True)
    # get default strategy file name
    strategy_local_path = os.path.join(ckpt_path, _strategy_dir)
    strategy_file = os.path.join(strategy_local_path, f"stratey{rank}.ckpt")
    # read ckpt name according to the ckpt path or return default name
    if get_name_from_file:
        rank_ckpts = glob.glob(os.path.join(ckpt_local_path, "*." + format))
        if not rank_ckpts:
            raise RuntimeError(f"{ckpt_local_path} has no .{format} ckpt file found")
        for checkpoint_file in rank_ckpts:
            if not os.path.isfile(checkpoint_file):
                ms.log.warning("{} is not a checkpoint file.".format(checkpoint_file))
                continue
            ckpt_file = checkpoint_file
    else:
        ckpt_file = os.path.join(ckpt_local_path, f"network{rank}" + "." + format)
    return ckpt_file, strategy_file

def save_rng_state():
    rng_state_dict = get_rng_tracer().get_state()
    rng_state_dict["default_generator"] = default_generator
    return rng_state_dict

def load_rng_state(param_dict):
    # set default rng tracer state
    target_state = {
        mode: param_dict.pop(mode) for mode in CANDIDATE_MODES if mode in param_dict
    }
    get_rng_tracer().set_state(target_state)
    # set default generator state
    default_generator_loaded = param_dict.pop("default_generator")
    set_rng_state(default_generator_loaded)

def _update_zero(params_dict, shard_info, allgather, param):
    tensor_concat = allgather(param)
    params_dict[param.name] = ms.Parameter(tensor_concat, name=param.name)
    shard_info[param.name]['opt_weight_shard_size'] = 0
    shard_info[param.name]['opt_weight_shard_step'] = 0

# pylint: disable=W0212
def save_pre_process(shard_info, model, optimizer, config):
    """ preprocess before saving, split qkv and handle pp embedding share """
    model_shard_info = shard_info["model"]
    optimizer_shard_info = shard_info["optimizer"]
    params_dict = model.parameters_dict() if optimizer is None else optimizer.parameters_dict()
    # ZeRO DP
    if optimizer is not None and hasattr(optimizer, "zero_level") and optimizer.zero_level in ["z1", "z2", "z3"]:
        allgather = P.AllGather(group=get_data_parallel_group(with_context_parallel=optimizer.with_context_parallel))
        for idx, param in enumerate(optimizer._parameters):
            if optimizer._status_splited[idx] or optimizer._parameter_splited[idx]:
                _update_zero(params_dict, optimizer_shard_info, allgather, optimizer.moments1[idx])
                _update_zero(params_dict, optimizer_shard_info, allgather, optimizer.moments2[idx])
            if optimizer.zero_level == "z3" and optimizer._parameter_splited[idx]:
                _update_zero(params_dict, model_shard_info, allgather, param)

    # process qkv/moe/pp-share
    for name, param in list(params_dict.items()):
        target_shard_info = model_shard_info if name in model_shard_info else optimizer_shard_info
        ### moe layer
        if config.moe_config is not None and config.moe_config.num_experts > 1 and "local_experts.0" in name:
            local_expert_num = config.moe_config.num_experts // get_expert_model_parallel_world_size()
            local_experts_list = []
            for idx in range(local_expert_num):
                local_expert_name = name.replace("local_experts.0", f"local_experts.{idx}")
                local_expert_param = params_dict.pop(local_expert_name).asnumpy()
                local_experts_list.append(local_expert_param)
                shard_dict = target_shard_info.pop(local_expert_name)
            local_experts_concat = np.stack(local_experts_list, axis=0)
            params_dict[name] = ms.Parameter(ms.Tensor(local_experts_concat))
            shard_dict['shape'] = local_experts_concat.shape
            shard_dict['shard'] = (get_expert_model_parallel_world_size(),) + shard_dict['shard']
            target_shard_info[name] = shard_dict

        ### handle pipeline head sharing
        if get_pipeline_model_parallel_world_size() == 1 and not config.untie_embeddings_and_output_weights:
            language_model_embedding = "language_model.embedding.word_embeddings.weight"
            language_model_head = "language_model.output_layer.weight"
            if language_model_embedding in name:
                new_name = name.replace(language_model_embedding, language_model_head)
                params_dict[new_name] = ms.Parameter(param, name=new_name)
                target_shard_info[new_name] = target_shard_info[name]

    return shard_info, params_dict

# pylint: disable=W0212
def load_post_process(config, params_dict, optimizer=None):
    """ load post processing, concat qkv """
    # process qkv
    for name, param in list(params_dict.items()):
        ### moe layer
        if config.moe_config is not None and config.moe_config.num_experts > 1 and "local_experts.0" in name:
            local_expert_num = config.moe_config.num_experts // get_expert_model_parallel_world_size()
            params_dict.pop(name)
            for shard_id in range(local_expert_num):
                new_name = name.replace("local_experts.0", f"local_experts.{shard_id}")
                params_dict[new_name] = ms.Parameter(ms.Tensor(param[shard_id]))

    ### ZeRO DP
    if optimizer is not None and hasattr(optimizer, "zero_level") and optimizer.zero_level in ["z1", "z2", "z3"]:
        shard_id = get_data_parallel_rank()
        split = P.Split(0, get_data_parallel_world_size())
        for idx, param in enumerate(optimizer._parameters):
            if optimizer._status_splited[idx] or optimizer._parameter_splited[idx]:
                # moments1
                moments1_name = optimizer.moments1[idx].name
                moments1 = params_dict[moments1_name]
                splited_tensor = split(moments1)[shard_id]
                params_dict[moments1_name] = ms.Parameter(splited_tensor, name=moments1_name)
                # moments2
                moments2_name = optimizer.moments2[idx].name
                moments2 = params_dict[moments2_name]
                splited_tensor = split(moments2)[shard_id]
                params_dict[moments2_name] = ms.Parameter(splited_tensor, name=moments2_name)
            if optimizer.zero_level == "z3" and optimizer._parameter_splited[idx]:
                # param
                cell_param = params_dict[param.name]
                splited_tensor = split(cell_param)[shard_id]
                params_dict[param.name] = ms.Parameter(splited_tensor, name=param.name)

    return params_dict

# pylint: disable=W0622
def save_checkpoint(config, model, optimizer=None, opt_param_scheduler=None, ckpt_path="./", format=_format,
                    only_save_strategy=False, **kwargs):
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
    # validator check
    validator.check_value_type("model", model, [nn.Cell], "save_checkpoint")
    validator.check_value_type("optimizer", optimizer, [nn.Cell, type(None)], "save_checkpoint")
    ckpt_file, strategy_file = get_checkpoint_name(ckpt_path, format=format)
    for key, _ in kwargs.items():
        logger.warning(f"The parameter {key} is not used in save_checkpoint.")
    logger.info(f"Saving model to {ckpt_path}")

    # generate sharded info
    shard_info = generate_state_dict(model, optimizer)
    shard_info, params_dict = save_pre_process(shard_info, model, optimizer, config)

    # saving
    save_strategy_file(shard_info, strategy_file)
    if not only_save_strategy:
        rng_state_dict = save_rng_state()
        append_dict = rng_state_dict.copy()
        if opt_param_scheduler is not None:
            opt_state_dict = opt_param_scheduler.state_dict()
            append_dict.update(opt_state_dict)
        ms.save_checkpoint(params_dict, ckpt_file, append_dict=append_dict, format=format)
    logger.info(f"ckpt saved")

# pylint: disable=W0622
def load_checkpoint(config, model, optimizer=None, opt_state_dict=None, ckpt_path="./", format=_format, **kwargs):
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
    logger.info(f"ckpt loading")
    src_ckpt_file, _ = get_checkpoint_name(ckpt_path, format=format, get_name_from_file=True)
    for key, _ in kwargs.items():
        logger.warning(f"The parameter {key} is not used in load_checkpoint.")

    target = optimizer if optimizer is not None else model
    param_dict = ms.load_checkpoint(src_ckpt_file, format=format)
    param_dict = load_post_process(config, param_dict, optimizer)

    load_rng_state(param_dict)
    if opt_state_dict is not None:
        opt_state_dict.load_state_dict(param_dict)
    param_not_load, ckpt_not_load = ms.load_param_into_net(target, param_dict)
    if param_not_load:
        logger.warning(f"param_not_load:{param_not_load}")
    if ckpt_not_load:
        logger.warning(f"ckpt_not_load:{ckpt_not_load}")
    logger.info(f"ckpt loaded")
