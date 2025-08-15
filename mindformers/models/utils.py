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
"""Check Model Input Config."""
import json
import os
from functools import wraps
from typing import Union
import mindspore.common.dtype as mstype
import mindspore as ms
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from ..tools.utils import get_predict_run_mode, is_pynative
from ..version_control import get_lazy_inline, get_predict_lazy_inline
from ..tools.logger import logger

# pylint: disable=W0212
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "mindspore_model.ckpt"
WEIGHTS_INDEX_NAME = "mindspore_model.ckpt.index.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
PROCESSOR_NAME = "processor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
MAX_INT32 = 2147483647

str_to_ms_type = {
    "float16": mstype.float16,
    "float32": mstype.float32,
    "bfloat16": mstype.bfloat16,
    "int8": mstype.int8
}


def convert_mstype(ms_type: str = "float16"):
    """Convert the string type to MindSpore type."""
    if isinstance(ms_type, mstype.Float):
        return ms_type
    ms_type = str(ms_type).lower()
    if ms_type == "float16":
        return mstype.float16
    if ms_type == "float32":
        return mstype.float32
    if ms_type == "bfloat16":
        return mstype.bfloat16
    if ms_type == "int8":
        return mstype.int8
    raise KeyError(f"Supported data type keywords include: "
                   f"[float16, float32, bfloat16, int8], but get {ms_type}")


def reverse_dict(d: dict):
    new_d = {}
    for k, v in d.items():
        if v in new_d:
            raise ValueError(f"Different keys in dict have same values.")
        new_d[v] = k
    return new_d


def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False


def check_fine_grain_interleave_valid(fine_grain_interleave, parallel_config):
    """Check the fine grain interleave condition"""
    if fine_grain_interleave is None or parallel_config is None:
        return False
    return fine_grain_interleave > 1 and parallel_config.model_parallel > 1


def check_use_3d_tensor_parallel_valid(config):
    """Check the use_3d_tensor_parallel condition"""
    use_3d_tensor_parallel = getattr(config, "use_3d_tensor_parallel", False)
    is_config_valid = config is not None and config.parallel_config is not None
    if not use_3d_tensor_parallel or not is_config_valid:
        return False
    if not config.use_flash_attention:
        raise ValueError(f"When the use_3d_tensor_parallel = True, the use_flash_attention must be True ")
    if config.parallel_config.get_ulysses_cp_num() > 1:
        raise ValueError(f"Currently, when the use_3d_tensor_parallel = True, "
                         "the cp_ds of the ulysses context parallel must be 1")
    if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
        raise ValueError(f"Currently, when the use_3d_tensor_parallel = True, the auto parallel is not supported")
    if config.moe_config is not None and config.moe_config.expert_num > 1:
        raise ValueError(f"Currently, when the use_3d_tensor_parallel = True, the MoE is not supported")
    if not config.parallel_config.use_seq_parallel:
        raise ValueError(f"Currently, when the use_3d_tensor_parallel = True, the use_seq_parallel must be True")
    if check_fine_grain_interleave_valid(config.fine_grain_interleave, config.parallel_config):
        raise ValueError("Currently, when the use_3d_tensor_parallel = True, "
                         "the fine_grain_interleave is not supported")
    tp_x = getattr(config, "tp_x", 1)
    tp_y = getattr(config, "tp_y", 1)
    tp_z = getattr(config, "tp_z", 1)
    model_parallel = config.parallel_config.model_parallel
    if model_parallel > 1 and tp_x * tp_y * tp_z != config.parallel_config.model_parallel:
        raise ValueError("tp_x * tp_y * tp_z should be equal to model_parallel, but got "
                         "tp_x={}, tp_y={}, tp_z={}, model_parallel={}.".format(tp_x, tp_y, tp_z, model_parallel))
    if model_parallel > 1:
        logger.info(f"use_3d_tensor_parallel is True, (tp_x, tp_y, tp_z): ({tp_x}, {tp_y}, {tp_z})")
        return True
    return False


def check_swap_enabled(swap_config):
    if isinstance(swap_config, dict):
        return swap_config["swap"]
    return swap_config.swap


def jit(func):
    """jit decorator."""

    @wraps(func)
    def decorator(*args, **kwargs):
        if not get_predict_run_mode():
            raise ValueError("Jit is only supported in predict mode now.")
        if is_pynative():
            return func(*args, **kwargs)
        return ms.jit(func, jit_level='O0', infer_boost='on')(*args, **kwargs)

    return decorator


def dict_from_json_file(json_file: Union[str, os.PathLike]):
    """method to read json."""
    if not os.path.exists(json_file):
        raise ValueError(
            f"{json_file} does not exist. Please check files in given path."
        )
    json_file = os.path.realpath(json_file)
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)

ms_type_to_str = reverse_dict(str_to_ms_type)

lazy_inline = get_lazy_inline
predict_lazy_inline = get_predict_lazy_inline
