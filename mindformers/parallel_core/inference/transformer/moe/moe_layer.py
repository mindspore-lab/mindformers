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

__all__ = [
    'MoESubmodules',
    'BaseMoELayer',
    'MoELayer'
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

from mindspore import Tensor, nn, mint, ops

from mindformers.parallel_core.inference.tensor_parallel.quantization import QuantizationConfig
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.inference.transformer.moe.router import TopKRouter
from mindformers.parallel_core.inference.tensor_parallel.mappings import (gather_from_model_parallel_region,
                                                                          reduce_from_model_parallel_region,
                                                                          reduce_scatter_to_model_parallel_region,)
from mindformers.parallel_core.inference.utils import divide
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs


@dataclass
class MoESubmodules:
    """
    MoE Layer Submodule spec:

    Args:
        experts (Union[ModuleSpec, type], optional): The module definition for experts.
            Defaults to None.
        shared_experts (Union[ModuleSpec, type], optional): The module definition for shared_experts.
            Defaults to None.
        token_dispatcher (Union[ModuleSpec, type], optional): The module definition for token dispatcher.
            Defaults to None.
    """

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None
    token_dispatcher: Union[ModuleSpec, type] = None


class BaseMoELayer(nn.Cell, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        layer_number (int): Number which indicates the index of this moe layer.
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.
    """

    def __init__(
            self,
            config: TransformerConfig,
            layer_number: int = None,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.tpdp_group = model_comm_pgs.tpdp
        self.ep_group = model_comm_pgs.moe_ep
        ep_size = self.ep_group.size

        if self.config.num_moe_experts % ep_size != 0:
            raise ValueError(f"The number of moe experts must be divisible by ep size, "
                             f"but got num experts: {self.config.num_moe_experts} and ep size: {ep_size}")
        self.num_local_experts = divide(self.config.num_moe_experts, ep_size)

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None

        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher = None

    @abstractmethod
    def construct(self, hidden_states, attn_unpadding_idx=None, ffn_padding_idx=None):
        """Forward method for the MoE layer."""
        raise NotImplementedError("MoELayer construct function not implemented.")


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
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ):
        self.submodules = submodules
        super().__init__(
            config=config, layer_number=layer_number, model_comm_pgs=model_comm_pgs
        )

        # Initialize router
        self.router = TopKRouter(config=self.config, model_comm_pgs=model_comm_pgs)

        # Initialize token dispatcher
        self.token_dispatcher = build_module(
            self.submodules.token_dispatcher,
            self.num_local_experts,
            config=self.config,
            model_comm_pgs=model_comm_pgs,
        )

        # Initialize experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            model_comm_pgs=model_comm_pgs,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config, model_comm_pgs=model_comm_pgs,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
            )

    def construct(self, hidden_states: Tensor, attn_unpadding_idx: Tensor = None, ffn_padding_idx: Tensor = None):
        """Construct MoELayer."""
        if self.config.attn_allgather:
            hidden_states = gather_from_model_parallel_region(hidden_states, self.tpdp_group, dim=0)
            hidden_states = ops.gather(hidden_states, attn_unpadding_idx, 0)

        # router
        expert_weight, routing_map = self.router(hidden_states)

        # token dispatch
        expert_weight, routing_map = self.token_dispatcher.dispatch_preprocess(expert_weight, routing_map)
        (
            dispatched_input,
            group_list,
            dispatched_outputs
        ) = self.token_dispatcher.token_dispatch(hidden_states, routing_map)

        # experts compute
        expert_output = self.experts(dispatched_input, group_list)

        # token combine
        output = self.token_dispatcher.token_combine(expert_output, expert_weight, *dispatched_outputs)

        if self.use_shared_expert:
            shared_expert_output = self.shared_experts(hidden_states)
            output = mint.add(output, shared_expert_output)

        if self.config.ffn_allreduce:
            output = reduce_from_model_parallel_region(output, self.tpdp_group)
        elif self.config.ffn_reduce_scatter:
            output = ops.gather(output, ffn_padding_idx, 0)
            output = reduce_scatter_to_model_parallel_region(output, self.tpdp_group)

        return output
