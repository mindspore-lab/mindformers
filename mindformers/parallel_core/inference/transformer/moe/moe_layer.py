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
"""moe layer for infer"""
from typing import Optional, Union

from mindspore import Tensor, nn, mint, ops
import mindspore.common.dtype as mstype
from mindspore.ops.auto_generate import (MoeInitRoutingV2,
                                         MoeTokenUnpermute)

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.inference.transformer.moe.router import TopKRouter
from mindformers.parallel_core.inference.tensor_parallel.mappings import reduce_from_model_parallel_region
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs

__all__ = [
    'MoESubmodules',
    'BaseMoELayer',
    'MoELayer'
]


class MoESubmodules:
    """
    MoE Layer Submodule spec:

    Args:
        experts (Union[ModuleSpec, type], optional): The module definition for experts.
            Defaults to None.
        shared_experts (Union[ModuleSpec, type], optional): The module definition for shared_experts.
            Defaults to None.
    """

    def __init__(self, experts: Union[ModuleSpec, type] = None, shared_experts: Union[ModuleSpec, type] = None):
        self.experts = experts
        self.shared_experts = shared_experts


class BaseMoELayer(nn.Cell):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        layer_number (int): Number which indicates the index of this moe layer.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super().__init__()
        self.config = config

        if self.config.expert_model_parallel_size > 1:
            raise NotImplementedError("For MoELayer, `expert_model_parallel_size` is not supported for now.")

        if config.moe_shared_expert_intermediate_size != 0:
            self.use_shared_expert = True
        else:
            self.use_shared_expert = False

        self.num_experts = config.num_moe_experts

        self.router = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        submodules (MoESubmodules): Submodules.
        layer_number (int): Number which indicates the index of this moe layer.
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: MoESubmodules = None,
            layer_number: int = None,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
    ):
        self.submodules = submodules
        super().__init__(config=config, layer_number=layer_number)

        if self.config.expert_model_parallel_size > 1:
            raise NotImplementedError("For MoELayer, `expert_model_parallel_size` is not supported for now.")

        # Initialize router
        self.router = TopKRouter(config=self.config)

        # Initialize token dispatcher
        # Note: It is not supported to initialize token dispatch currently.

        # Initialize experts
        self.experts = build_module(self.submodules.experts, self.num_experts, self.config)

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)

        self.cast = ops.Cast()

        self.moe_init_routing_v2 = MoeInitRoutingV2()
        self.moe_token_unpermute = MoeTokenUnpermute()
        self.tp_group = model_comm_pgs.tp

    def construct(self, hidden_states: Tensor):
        """Construct MoELayer."""
        # [1, B * S, H] -> [T, H]
        input_tensor_shape = hidden_states.shape
        hidden_states = hidden_states.reshape((-1, self.config.hidden_size))

        # router
        expert_weight, routing_map = self.router(hidden_states)

        # token_permutation
        dispatched_input, tokens_per_expert, group_list, _ = \
            self.moe_init_routing_v2(
                hidden_states,
                routing_map,
                active_num=0,
                expert_capacity=0,
                expert_num=self.config.num_moe_experts,
                drop_pad_mode=0,
                expert_tokens_count_or_cumsum_flag=2,
                expert_tokens_before_capacity_flag=True
            )

        group_list = self.cast(group_list, mstype.int64)
        expert_output = self.experts(dispatched_input, group_list)

        moe_output = self.moe_token_unpermute(permuted_tokens=expert_output,
                                              sorted_indices=tokens_per_expert,
                                              probs=expert_weight,
                                              padded_mode=False,
                                              restore_shape=None)

        output = reduce_from_model_parallel_region(moe_output, self.tp_group)

        if self.use_shared_expert:
            output = mint.add(output, self.shared_experts(hidden_states))

        output = output.reshape(input_tensor_shape)
        return output
