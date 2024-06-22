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
import re
import numpy as np
import mindspore.common.dtype as mstype
from ..version_control import get_lazy_inline, get_predict_lazy_inline
from ..tools.logger import logger

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "mindspore_model.ckpt"
WEIGHTS_INDEX_NAME = "mindspore_model.ckpt.index.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
PROCESSOR_NAME = "processor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME

str_to_ms_type = {
    "float16": mstype.float16,
    "float32": mstype.float32,
    "bfloat16": mstype.bfloat16
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
    raise KeyError(f"Supported data type keywords include: "
                   f"[float16, float32, bfloat16], but get {ms_type}")


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

def _check_layer_rule(layer_id, accu_layer, stage_layer, pp_id):
    diff = stage_layer[pp_id] - (accu_layer[pp_id + 1] -accu_layer[pp_id]) // 2
    full_range = diff * 2 if diff > 0 else 0
    if layer_id < full_range:
        return True
    if layer_id % 2 == 0 and (layer_id - accu_layer[pp_id]) //2 < stage_layer[pp_id]:
        return True
    return False

# pylint: disable=W0212
def _set_pattern_recompute(layer, p_list, add_prim_attr=False, info='layer'):
    """Set recompute pattern to layer."""
    if not p_list:
        return
    p = p_list.pop(0)
    if p_list:
        for name, cell in layer._cells.items():
            if re.fullmatch(p, name):
                _set_pattern_recompute(cell, p_list, add_prim_attr, info + f'.{name}')
    else:
        for attr in dir(layer):
            if re.fullmatch(p, attr):
                operator = getattr(layer, attr)
                if add_prim_attr:
                    operator.add_prim_attr("recompute_comm_op", True)
                    logger.info(f"Set select comm recompute: {info}.{attr}")
                elif hasattr(operator, "recompute"):
                    operator.recompute()
                    logger.info(f"Set select recompute: {info}.{attr}")
        for name, cell in layer._cells.items():
            if re.fullmatch(p, name):
                if not add_prim_attr:
                    cell.recompute()
                    logger.info(f"Set select recompute: {info}.{name}")
    p_list.insert(0, p)


def _set_select_recompute(layer, select_recompute, pp_id, layer_id, layer_list, default_patterns, add_prim_attr=False):
    """Set select recompute."""
    layer_list_mod = np.insert(layer_list, 0, 0)
    if isinstance(select_recompute, bool):
        if select_recompute:
            for p in default_patterns:
                _set_pattern_recompute(layer, p.split(r'\.'), add_prim_attr, f'layer_{layer_id}')
    elif isinstance(select_recompute, (list, tuple)):
        if all(isinstance(item, int) for item in select_recompute):
            if layer_id < layer_list_mod[pp_id] + select_recompute[pp_id]:
                for p in default_patterns:
                    _set_pattern_recompute(layer, p.split(r'\.'), add_prim_attr, f'layer_{layer_id}')
        elif all(isinstance(item, str) for item in select_recompute):
            for p in select_recompute:
                _set_pattern_recompute(layer, p.split(r'\.'), add_prim_attr, f'layer_{layer_id}')
        else:
            raise ValueError(f"Illegal input format for select_recompute: {select_recompute}")
    elif isinstance(select_recompute, dict):
        for k, v in select_recompute.items():
            if not all(isinstance(item, int) for item in v):
                raise ValueError(f"Illegal input format for select_recompute: {k}: {v}")
            if layer_id < layer_list_mod[pp_id] + v[pp_id]:
                _set_pattern_recompute(layer, k.split(r'\.'), add_prim_attr, f'layer_{layer_id}')


def set_layer_stage_recompute(layer, layer_id, offset, parallel_config, n_layers):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            layer(Cell) - Represents the transformer block
            parallel_config(dict) - Parallel Config
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(Union[int, List[int]]) - Means the layer_index needs a offset, if there are other modules in the net.
            n_layers(int) - The total layers used for the model.
    """
    pp = parallel_config.pipeline_stage
    stage_layers_list = np.array([n_layers // pp] * pp) + np.array(offset)
    layer_list = np.array([np.sum(stage_layers_list[:i + 1]) for i in range(len(stage_layers_list))])
    if isinstance(offset, (list, tuple)):
        if len(offset) != pp:
            raise ValueError(f"The length of `offset` {len(offset)} do not match `pipeline stage` {pp}.")
        pp_id = int(np.sum(layer_list < layer_id + 1))
        offset_layer = offset[0]
    elif isinstance(offset, int):
        offset_layer = offset
        pp_dis = max(int((n_layers + 1) / pp), 1)
        pp_id = min((layer_id + offset_layer) // pp_dis, pp - 1)
    else:
        raise TypeError(f"`offset` must be `int` or list/tuple of `int`, but got {type(offset)}.")

    layer.pipeline_stage = pp_id

    # Used for optimizer's fusion tag
    dis = max(int((n_layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if pp > 1:
        layer.set_comm_fusion(2)
    else:
        layer.set_comm_fusion(int((layer_id + offset_layer) / dis) + 1)
    if isinstance(parallel_config.recompute, bool):
        if parallel_config.recompute:
            layer.recompute()
            return
    select_recompute = parallel_config.recompute.select_recompute
    select_comm_recompute = parallel_config.recompute.select_comm_recompute
    if parallel_config.recompute.recompute:
        if isinstance(parallel_config.recompute.recompute, bool):
            layer.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)
        elif isinstance(parallel_config.recompute.recompute, (list, tuple)):
            layer_list_mod = np.insert(layer_list, 0, 0)
            if _check_layer_rule(layer_id, layer_list_mod, parallel_config.recompute.recompute, pp_id):
                layer.recompute()
                logger.info(f"Set layer recompute: layer_{layer_id}")
            else:
                default_patterns = [r'feed_forward\.mul', r'feed_forward\.w1\.activation\.silu']
                default_comm_patterns = [r'.*\.norm']
                _set_select_recompute(layer, select_recompute, pp_id, layer_id, layer_list, default_patterns, False)
                _set_select_recompute(layer, select_comm_recompute, pp_id, layer_id, layer_list, default_comm_patterns,
                                      True)
        else:
            raise ValueError(f"reompute.recompute should be bool/list/tuple, but got: "
                             f"{type(parallel_config.recompute.recompute)} ({parallel_config.recompute.recompute})")
    else:
        default_patterns = [r'feed_forward\.mul', r'feed_forward\.w1\.activation\.silu']
        default_comm_patterns = [r'.*\.norm']
        _set_select_recompute(layer, select_recompute, pp_id, layer_id, layer_list, default_patterns, False)
        _set_select_recompute(layer, select_comm_recompute, pp_id, layer_id, layer_list, default_comm_patterns, True)


def check_fine_grain_interleave_valid(fine_grain_interleave, parallel_config):
    """Check the fine grain interleave condition"""
    if fine_grain_interleave is None or parallel_config is None:
        return False
    return fine_grain_interleave > 1 and parallel_config.model_parallel > 1


ms_type_to_str = reverse_dict(str_to_ms_type)

lazy_inline = get_lazy_inline
predict_lazy_inline = get_predict_lazy_inline
