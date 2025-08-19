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
"""MoE module spec."""
from typing import Optional

from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.parallel_core.inference.transformer.mlp import MLPSubmodules
from mindformers.parallel_core.inference.transformer.moe.moe_layer import MoELayer, MoESubmodules
from mindformers.parallel_core.inference.transformer.moe.experts import GroupedMLP
from mindformers.parallel_core.inference.transformer.moe.shared_experts import SharedExpertMLP
from mindformers.parallel_core.inference.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from mindformers.parallel_core.inference.tensor_parallel.layers import (
    MergedColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from mindformers.parallel_core.inference.tensor_parallel.gemm_layers import (
    ColumnParallelGroupedLinear,
    RowParallelGroupedLinear
)


def get_moe_module_spec(
        num_experts: Optional[int] = None,
        moe_grouped_gemm: Optional[bool] = True,
        use_alltoall: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    if not num_experts:
        raise ValueError(f"Using MoE module, num_experts must be int, but num_experts get {num_experts}.")

    if not moe_grouped_gemm:
        raise NotImplementedError("moe_grouped_gemm = 'False' is not supported now.")

    # experts spec
    ## use legacy GroupedMLP
    expert_module = GroupedMLP
    expert_submodule = MLPSubmodules(
        linear_fc1=ColumnParallelGroupedLinear,
        linear_fc2=RowParallelGroupedLinear,
    )

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # shared experts spec
    shared_expert_module = SharedExpertMLP
    shared_expert_submodule = MLPSubmodules(
        # When using AlltoAll, shared experts use unsplit linear
        linear_fc1=ReplicatedLinear if use_alltoall else MergedColumnParallelLinear,
        linear_fc2=ReplicatedLinear if use_alltoall else RowParallelLinear,
    )
    shared_experts = ModuleSpec(module=shared_expert_module, params={"gate": False}, submodules=shared_expert_submodule)

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(
            experts=experts,
            shared_experts=shared_experts,
            token_dispatcher=MoEAlltoAllTokenDispatcher if use_alltoall else MoEAllGatherTokenDispatcher,
        )
    )

    return moe_module_spec
