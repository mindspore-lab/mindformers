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
"""MoE Module Spec."""
from typing import Optional

from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.parallel_core.training_graph.transformer.mlp import MLPSubmodules
from mindformers.parallel_core.training_graph.transformer.moe.shared_experts import SharedExpertMLP
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.parallel_core.training_graph.transformer.moe.moe_layer import MoELayer, MoESubmodules
from mindformers.parallel_core.training_graph.transformer.moe.ffn import FFNGroupedGEMM


def get_moe_module_spec(
        num_experts: Optional[int] = None,
        moe_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    if num_experts is None:
        raise ValueError("num_experts cannot be None.")

    # experts spec
    if not moe_grouped_gemm:
        raise NotImplementedError("moe_grouped_gemm = 'False' is not supported now.")

    moe_module_spec = ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(
            experts=FFNGroupedGEMM,
            shared_experts=ModuleSpec(
                module=SharedExpertMLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear
                ),
            )
        )
    )
    return moe_module_spec
