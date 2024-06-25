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
"""utils"""
import inspect
import mindspore.common.dtype as mstype
from mindspore.communication import get_group_size
from mindspore.nn.optim.optimizer import Optimizer
from mindformers.experimental.distri_cores.transformer import Module
from mindformers.experimental.distri_cores.create_comm import (
    get_pp_rank,
    get_pp_world_size,
)

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def get_layer_input_signatures(func):
    """except for 'input_ids', get all of input signatures for layer's construct."""
    params = inspect.signature(func).parameters
    # remove 'input_ids' argument
    layer_input_signatures = list(params.keys())[1:]
    if not layer_input_signatures:
        layer_input_signatures = None
    return layer_input_signatures


def add_attr_for_shared_weight(layer, weight_name='weight'):
    """ add 'share' attr for embedding or head layer weight """
    cur_layer = layer
    param_name = weight_name
    if '.' in weight_name:
        sub_module = [item for item in weight_name.split('.')]
        param_name = sub_module[-1]
        sub_module = sub_module[:-1]
        for next_layer in sub_module:
            cur_layer = getattr(cur_layer, next_layer)
    if hasattr(cur_layer, param_name):
        param_instance = getattr(cur_layer, param_name)
        if param_instance is not None:
            setattr(param_instance, 'share', True)
        else:
            print(f"[WARNING]For 'add_attr_for_shared_weight' function, "
                  f"class '{type(layer).__name__}' weight is None, so the 'share' attr cannot be added.")
    else:
        print(f"[WARNING]For 'add_attr_for_shared_weight' function, "
              f"class '{type(layer).__name__}' have no weight for adding 'share' attr")


def convert_mstype(ms_type: str = "float16"):
    """Convert the string type to MindSpore type."""
    if isinstance(ms_type, mstype.Float):
        return ms_type
    if ms_type == "float16":
        return mstype.float16
    if ms_type == "float32":
        return mstype.float32
    if ms_type == "bfloat16":
        return mstype.bfloat16
    raise KeyError(f"Supported data type keywords include: "
                   f"[float16, float32, bfloat16], but get {ms_type}")

def get_default_dict_for_optimizer(optimizer, model_sharded_state_dict):
    """get default sharded state dict for the optimizer with models' shard and no opt shard"""
    state_dict = {}
    for model_param in optimizer.parameters:
        model_name = model_param.name
        if model_name in model_sharded_state_dict and 'shard' in model_sharded_state_dict[model_name]:
            shard = list(model_sharded_state_dict[model_name]['shard'])
        else:
            raise Exception(f"the input dict has no shard info for '{model_name}'.")

        for optim_param in optimizer.get_parameters():
            optim_name = optim_param.name
            if optim_name.endswith(model_name) and optim_name != model_name:
                state_dict[optim_name] = {'shape': model_param.shape, 'shard': tuple(shard),
                                          'opt_weight_shard_step': 0, 'opt_weight_shard_size': -1}
    return state_dict

def generate_state_dict(network: Module, optimizer: Optimizer):
    r"""
    Generete the sharded stated dict for the network and optimizer.

    The `network` should be of type Module which has inherited or overridden method sharded_state_dict,
    The `optimizer` is the corresponding optimizer.

    Args:
        network (Module): the integral model to be used
        optimizer (Optimizer): the corresponding optimizer

    Returns:
        Dict, which contains the necessary sharded info for checkpoint transformation, e.g. 'total_rank',
    'stage_rank_size', 'stage' for pipeline, etc.

    Supported Platforms:
        ``Ascend``
    """
    try:
        pp_size = get_pp_world_size()
        pp_rank = get_pp_rank()
    except AssertionError:
        pp_size = 1
        pp_rank = 0
    state_dict = {
        'total_rank': get_group_size(),
        'stage_rank_size': get_group_size() // pp_size,
        'stage': pp_rank,
    }
    state_dict['model'] = network.sharded_state_dict()
    state_dict['optimizer'] = {}
    if optimizer is not None:
        if hasattr(optimizer, 'sharded_state_dict'):
            state_dict['optimizer'] = optimizer.sharded_state_dict(state_dict['model'])
        else:
            print(f"The optimizer {type(optimizer).__name__} has no sharded_state_dict overridden")
            state_dict['optimizer'] = get_default_dict_for_optimizer(optimizer, state_dict['model'])
    return state_dict

def save_strategy_file(state_dict, strategy_file_name):
    r"""
    Save the strategy file according to the state_dict and strategy_file_name

    Args:
        state_dict (Dict): dict with sharding metainfo
        strategy_file_name (String): the name of the target saving file

    Supported Platforms:
        ``Ascend``
    """
    import os
    import stat
    from mindspore import log as logger
    from mindspore.train.node_strategy_pb2 import ParallelStrategyMap as ckpt_strategy
    stra = ckpt_strategy()

    # pylint: disable=W0612
    total_rank = state_dict["total_rank"]
    stage_rank_size = state_dict["stage_rank_size"]
    stage = state_dict["stage"]
    model_param = state_dict["model"]
    optimizer_param = state_dict["optimizer"]
    stra.current_stage = 0
    model_param.update(optimizer_param)
    for name, item in model_param.items():
        if "shard" not in item or "shape" not in item:
            continue
        opt_weight_shard_step = item["opt_weight_shard_step"] if "opt_weight_shard_step" in item.keys() else 0
        opt_weight_shard_size = item["opt_weight_shard_size"] if "opt_weight_shard_size" in item.keys() else -1
        strategy_item = stra.parallel_strategy_item.add()
        strategy_item.node_name = name
        parallel_strategys = strategy_item.parallel_strategys
        parallel_strategys.stage = stage
        shard = item["shard"]
        shape = item["shape"]
        parallel_strategy = parallel_strategys.parallel_strategy.add()
        shard_mul = 1
        for ele in shard:
            parallel_strategy.dim.append(ele)
            shard_mul = shard_mul * ele
        layout_item = stra.parallel_layout_item.add()
        layout_item.param_name = name
        parallel_layouts = layout_item.parallel_layouts
        parallel_layouts.field = 0
        parallel_layouts.opt_weight_shard_step = opt_weight_shard_step
        parallel_layouts.opt_weight_shard_size = opt_weight_shard_size
        dev_matrix = parallel_layouts.dev_matrix.add()
        repeat_calc_num = 1
        if stage_rank_size == shard_mul:
            repeat_calc_num = 1
        elif stage_rank_size % shard_mul == 0:
            repeat_calc_num = stage_rank_size // shard_mul
        else:
            raise ValueError(f"For {name}, the shard{shard} requires {shard_mul} devices, "
                             f"but the device number of this stage is {stage_rank_size}, "
                             f"it can not be divisible by {shard_mul}")
        if repeat_calc_num != 1:
            dev_matrix.dim.append(repeat_calc_num)
        for ele in shard:
            dev_matrix.dim.append(ele)
        tensor_map = parallel_layouts.tensor_map.add()
        shape_len = len(shape)
        index = shape_len - 1
        for _ in range(shape_len):
            tensor_map.dim.append(index)
            index = index - 1
        param_split_shape = parallel_layouts.param_split_shape.add()
        for ele in shape:
            param_split_shape.dim.append(ele)

    try:
        if os.path.exists(strategy_file_name):
            os.chmod(strategy_file_name, stat.S_IWUSR)
        if "/" in strategy_file_name:
            real_path = os.path.abspath(strategy_file_name[:strategy_file_name.rfind("/")])
            os.makedirs(real_path, exist_ok=True)
        with open(strategy_file_name, "wb") as f:
            f.write(stra.SerializeToString())
            os.chmod(strategy_file_name, stat.S_IRUSR)

    except BaseException as e:
        logger.critical(f"Failed to save the checkpoint file {strategy_file_name}. Maybe don't have "
                        "the permission to write files, or the disk space is insufficient and so on.")
        raise e
