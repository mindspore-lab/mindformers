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
"""Transformer MoE Layer."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

from mindspore import nn
from mindspore.context import ParallelMode
from mindspore.ops.auto_generate import AddExt, Reshape, Shape, Transpose
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.parallel_core.training_graph.transformer.moe.router import TopKRouter
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import TransformerConfig


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(nn.Cell, ABC):
    """
    Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_number = layer_number

        self.expert_parallel_size = config.expert_model_parallel_size
        if self.expert_parallel_size <= 0:
            raise ValueError("Expected non-negative expert parallel size")
        if config.num_moe_experts % self.expert_parallel_size != 0:
            raise ValueError(
                f"Expected num_moe_experts {config.num_moe_experts} to be divisible"
                f" by expert_model_parallel_size {config.expert_model_parallel_size}.")

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

    @abstractmethod
    def construct(self, x, extra_loss=0., seq_chunk=None):
        r"""Forward process of the moe layer"""

    @abstractmethod
    def shard(self, config: TransformerConfig):
        r"""set parallel strategy"""


class MoELayer(BaseMoELayer):
    """
    The mixture of experts (MoE) implementation (using GMM op instead of BMM op used by MOE module).
    The implementation includes a router and a FeedForward layer.
    The router dispatches tokens to experts in FeedForward, then FeedForward does computation, and the final output is
    obtained by multiplying FeedForward's output and router's combine weight.

    Args:
        config (TransformerConfig): The configuration of MoE (Mixture of Expert). Please see `TransformerConfig`.
        submodules (MoESubmodules): The submodules of MoE. Please see `MoESubmodules`.
        layer_number (int): The layer number of the MoE layer. Default: None.
    """
    def __init__(
            self,
            config: TransformerConfig,
            submodules: Optional[MoESubmodules] = None,
            layer_number: Optional[int] = None,
    ):
        # reversed arg
        super().__init__(config=config, layer_number=layer_number)
        self.hidden_size = config.hidden_size
        self.use_seq_parallel = config.sequence_parallel
        self.dp = config.data_parallel_size * config.tensor_model_parallel_size

        # ops
        self.add = AddExt()
        if self.shared_expert_overlap:
            self.add.add_prim_attr("parallel_branch", 1)
        self.reshape = Reshape()
        self.shape = Shape()
        self.add_loss = AddExt()
        self.transpose = Transpose()
        self.transpose2 = Transpose()

        # router
        self.router = TopKRouter(config)

        # experts
        self.experts = build_module(submodules.experts, config=config)

        # shared_experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                submodules.shared_experts,
                config=config
            )

        # sharding
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        elif _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard(config)

    def construct(self, x, extra_loss=0., seq_chunk=None):
        """Construct function of the MoELayer."""
        x = self.transpose(x, (1, 0, 2))

        origin_shape = self.shape(x)
        # The shape change is: (dp, N, h) <-- (B*S, h)
        x_reshaped = self.reshape(x, (self.dp, -1, self.hidden_size))

        # 1. router
        expert_index, router_coeff, router_aux_loss = self.router(x_reshaped)

        # 2. permute + experts + unpermute
        experts_output = self.experts(x_reshaped, router_coeff, expert_index)
        experts_output = self.reshape(experts_output, origin_shape)

        # 3. shared experts
        if self.use_shared_expert:
            shared_experts_output, _ = self.shared_experts(x)
            experts_output = self.add(experts_output, shared_experts_output)

        # BSH -> SBH
        experts_output = self.transpose2(experts_output, (1, 0, 2))

        extra_loss = self.add_loss(extra_loss, router_aux_loss)
        return experts_output, None, extra_loss

    def shard(self, config: TransformerConfig):
        """Set parallel strategy."""
        dp = config.data_parallel_size
        mp = config.tensor_model_parallel_size
        self.transpose.shard(((1, dp, mp),))
        self.transpose2.shard(((dp, 1, mp),))
        if self.use_seq_parallel:
            self.add.shard(((dp, mp, 1), (dp, mp, 1)))
        else:
            self.add.shard(((dp, 1, 1), (dp, 1, 1)))
        self.add_loss.shard(((1,), (1,)))
