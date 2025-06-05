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
"""run moe in infer mode"""
import numpy as np

import mindspore as ms
from mindspore import Tensor
import mindspore.common.dtype as mstype

from mindformers.models.llama import LlamaConfig
from mindformers.modules.transformer.moe import MoEConfig
from mindformers.parallel_core.transformer_config import TransformerConfig

from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_moe.utils import (
    NewMoENet,
    OldMoENet,
    get_init_params,
    convert_weight_name
)

jit_level = "O0"
infer_boost = "on"
ms.set_context(device_target="Ascend",
               deterministic="ON",
               mode=ms.GRAPH_MODE,
               jit_config={"jit_level": jit_level, "infer_boost": infer_boost})

seed_value = 42
ms.set_seed(seed_value)
np.random.seed(seed_value)


BATCH_SIZE = 2
SEQ_LENGTH = 2
HIDDEN_SIZE = 32
n_shared_experts = 1
routed_scaling_factor = 2.5
n_routed_experts = 8
hidden_act = "silu"
num_experts_per_tok = 2
moe_intermediate_size = 8
n_group = 2
topk_group = 2

def set_model_config():
    """Set model config for moe ut test."""
    config = LlamaConfig()
    config.batch_size = 2
    config.seq_length = 2
    config.hidden_size = HIDDEN_SIZE
    config.param_init_type = mstype.bfloat16
    config.compute_dtype = mstype.bfloat16
    config.router_dense_type = "bfloat16"
    config.hidden_act = 'silu'
    config.mlp_has_bias = False
    config.mlp_has_gate = True
    config.ffn_concat = True
    config.moe_config = MoEConfig(expert_num=n_routed_experts, num_experts_chosen=num_experts_per_tok,
                                  shared_expert_num=n_shared_experts, routed_scaling_factor=routed_scaling_factor,
                                  moe_intermediate_size=moe_intermediate_size, topk_group=topk_group,
                                  n_group=n_group)
    return config


if __name__ == '__main__':
    model_config = set_model_config()
    transformer_config = TransformerConfig(
        num_layers=1,
        num_attention_heads=8,
        hidden_size=HIDDEN_SIZE,
        hidden_act="silu",
        num_moe_experts=n_routed_experts,
        moe_router_topk=num_experts_per_tok,
        shared_expert_num=num_experts_per_tok,
        moe_router_topk_scaling_factor=routed_scaling_factor,
        moe_shared_expert_intermediate_size=moe_intermediate_size,
        moe_router_group_topk=topk_group,
        moe_router_num_groups=n_group,
        add_bias_linear=False,
        gated_linear_unit=True,
        compute_dtype="bfloat16",
        params_dtype="bfloat16",
        moe_router_dtype="bfloat16",
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
    )

    hidden_size = model_config.hidden_size
    params = get_init_params(hidden_size, n_routed_experts, moe_intermediate_size)

    input_x = Tensor(params.pop("input"), dtype=mstype.bfloat16)

    new_weight = params

    new_net = NewMoENet(config=transformer_config)
    ms.load_param_into_net(new_net, new_weight)
    old_net = OldMoENet(config=model_config)
    old_weight = convert_weight_name(new_weight)
    ms.load_param_into_net(old_net, new_weight)

    new_moe_output = new_net(input_x)
    old_moe_output = old_net(input_x)

    ret = np.array_equal(new_moe_output.asnumpy(), old_moe_output.asnumpy())

    assert ret
