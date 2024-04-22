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
import numpy as np
import mindspore.common.dtype as mstype
from ..version_control import get_cell_reuse

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "mindspore_model.ckpt"
WEIGHTS_INDEX_NAME = "mindspore_model.ckpt.index.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
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


def check_recompute_rule(select_recompute, pp_id, layer_id, layer_list):
    if isinstance(select_recompute, bool):
        return select_recompute
    if isinstance(select_recompute, (list, tuple)):
        layer_list = np.insert(layer_list, 0, 0)
        if layer_id < layer_list[pp_id] + select_recompute[pp_id]:
            return True
    return False


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
    if parallel_config.recompute.recompute and not parallel_config.recompute.select_recompute:
        layer.recompute(
            recompute_slice_activation=parallel_config.recompute.recompute_slice_activation
        )
    else:
        if check_recompute_rule(parallel_config.recompute.select_comm_recompute, pp_id, layer_id, layer_list):
            if not layer.attention_norm.self_define:
                layer.attention_norm.norm.add_prim_attr("recompute_comm_op", True)
                layer.ffn_norm.norm.add_prim_attr("recompute_comm_op", True)
        if check_recompute_rule(parallel_config.recompute.select_recompute, pp_id, layer_id, layer_list):
            layer.feed_forward.mul.recompute()
            layer.feed_forward.w1.activation.silu.recompute()


ms_type_to_str = reverse_dict(str_to_ms_type)

cell_reuse = get_cell_reuse
