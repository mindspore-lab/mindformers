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
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank, get_group_size, create_group
from mindspore.ops.auto_generate import AddExt, Reshape, Shape, Transpose
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.parallel_core.training_graph.device_matrix import layout, layout_moe
from mindformers.parallel_core.training_graph.transformer.moe.router import TopKRouter
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.transformer.moe.expert_mapping import ExpertDynamicRelocation


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

        if self.config.print_expert_load or self.config.enable_expert_relocation:
            self.expert_load_history = Parameter(
                Tensor(np.zeros(self.config.num_moe_experts), ms.float32), requires_grad=False)
            self.expert_load_history_cnt = Parameter(
                Tensor(0, dtype=ms.int32), requires_grad=False)
            self.expert_mapping = Parameter(Tensor(list(range(
                self.config.num_moe_experts)), ms.int32), requires_grad=False)
            self.expert_relocation_optimizer = ExpertDynamicRelocation()

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
        layout_moe.init_layout(config)
        self.hidden_size = config.hidden_size
        self.use_seq_parallel = config.sequence_parallel
        self.dp = (
            config.data_parallel_size *
            config.tensor_model_parallel_size *
            config.context_parallel_size
        )
        self.tp = config.tensor_model_parallel_size

        if self.config.print_expert_load or self.config.enable_expert_relocation:
            self.dp_group = self._dp_group()
            self.ep = config.expert_model_parallel_size
            self.expert_num = config.num_moe_experts
            self.num_local_experts = self.expert_num // self.ep

        # ops
        self.add = AddExt()
        if self.shared_expert_overlap:
            self.add.add_prim_attr("parallel_branch", 1)
        self.reshape = Reshape()
        self.shape = Shape()
        self.add_loss = AddExt()
        self.transpose = Transpose()
        self.transpose2 = Transpose()

        # check_rules
        if self.tp > 1 and not self.use_seq_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

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

    def _dp_group(self):
        """Create MoE data parallel group across DP."""
        rank_id = get_rank()
        world_size = get_group_size()
        dp_group_id = rank_id // self.dp

        start_rank = dp_group_id * self.dp
        end_rank = min(start_rank + self.dp, world_size)

        rank_list = [i for i in range(start_rank, end_rank)]

        rank_list_str = "-".join([str(i) for i in rank_list])
        hashed = hashlib.sha256(rank_list_str.encode()).hexdigest()[:48]
        dp_group_name = str(hashed)
        create_group(dp_group_name, rank_list)
        return dp_group_name

    def construct(self, hidden_states, extra_loss=0., seq_chunk=None):
        """Construct function of the MoELayer."""
        x = self.transpose(hidden_states, (1, 0, 2))

        origin_shape = self.shape(x)
        # The shape change is: (dp, N, h) <-- (B*S, h)
        x_reshaped = self.reshape(x, (self.dp, -1, self.hidden_size))

        # 1. router
        expert_index, router_coeff, router_aux_loss = self.router(x_reshaped)

        if self.config.enable_expert_relocation:
            expert_index = self.expert_mapping[expert_index]

        # 2. permute + experts + unpermute
        experts_output = self.experts(x_reshaped, router_coeff, expert_index)
        experts_output = self.reshape(experts_output, origin_shape)

        # BSH -> SBH
        experts_output = self.transpose2(experts_output, (1, 0, 2))

        # 3. shared experts
        if self.use_shared_expert:
            shared_experts_output, _ = self.shared_experts(hidden_states)
            experts_output = self.add(experts_output, shared_experts_output)

        extra_loss = self.add_loss(extra_loss, router_aux_loss)
        return experts_output, None, extra_loss

    def shard(self, config: TransformerConfig):
        """Set parallel strategy."""
        self.transpose.shard((layout("cp", "dp", "None"),))
        if self.use_seq_parallel:
            self.transpose2.shard((layout("dp", "cp_tp", "None"),))
            self.add.shard((layout("cp_tp", "dp", "None"),
                            layout("cp_tp", "dp", "None")))
        else:
            self.transpose2.shard((layout("dp", "cp", "None"),))
            self.add.shard((layout("cp", "dp", "None"),
                            layout("cp", "dp", "None")))

    def update_expert_load_history(self, num_tokens_per_expert):
        """
        Update expert load history based on token distribution.

        Args:
            num_tokens_per_expert: Array containing number of tokens assigned to each expert.
        """
        if self.config.enable_expert_relocation:
            # map back the num_tokens to the original order of experts
            num_tokens_per_expert_origin = num_tokens_per_expert[self.expert_mapping]
        else:
            num_tokens_per_expert_origin = num_tokens_per_expert
        # update expert load
        expert_load_history_cnt_new = self.expert_load_history_cnt + 1
        expert_load_new = ops.cast((self.expert_load_history * self.expert_load_history_cnt
                                    + num_tokens_per_expert_origin) / (expert_load_history_cnt_new), ms.float32)

        self.expert_load_history.set_data(expert_load_new)
        if self.config.print_expert_load:
            expert_load_history_cnt_new = ops.minimum(
                expert_load_history_cnt_new, Tensor(100, ms.int32))

        self.expert_load_history_cnt.set_data(expert_load_history_cnt_new)

    def gather_expert_load_data_parallel(self, num_tokens_per_expert):
        if self.dp > 1:
            num_tokens_per_expert_overall = ops.AllReduce(
                group=self.dp_group)(num_tokens_per_expert)
            num_tokens_per_expert_overall = num_tokens_per_expert_overall // self.dp
        else:
            num_tokens_per_expert_overall = num_tokens_per_expert.clone()
        return num_tokens_per_expert_overall

    def initialize_expert_relocation_dispatcher(self, is_triggered_restore=False):
        """
        Initialize expert relocation dispatcher for load balancing.

        Args:
            is_triggered_restore: Whether to restore original expert mapping.

        Returns:
            tuple: (device_expert_mapping, new_local_expert_sorted_indices)
        """
        self.expert_load_history.set_data(
            self.gather_expert_load_data_parallel(self.expert_load_history))
        expert_mapping = self.expert_mapping.asnumpy()

        ep_devices = list(range(self.ep))
        expert_load = dict(
            zip(range(self.config.num_moe_experts), self.expert_load_history.asnumpy()))

        if is_triggered_restore:
            experts_per_device = self.config.num_moe_experts // self.ep
            expert_device_mapping = {}
            for current_pos in range(self.config.num_moe_experts):
                orig_expert_id = list(expert_mapping).index(current_pos)
                device_id = orig_expert_id // experts_per_device
                expert_device_mapping[current_pos] = [device_id]
            device_expert_mapping = {device_id: []
                                     for device_id in range(self.ep)}
            for pos, dev_id_list in expert_device_mapping.items():
                dev_id = dev_id_list[0]
                device_expert_mapping[dev_id].append(pos)
            for dev_id in device_expert_mapping:
                device_expert_mapping[dev_id].sort()
        else:
            expert_device_mapping, device_expert_mapping = self.expert_relocation_optimizer.expert_relocation_greedy(
                expert_load, ep_devices)

        # turn expert_device_mapping into expert_index mapping
        device_expert_idx = [set() for i in range(self.ep)]
        device_full_expert_idx = {i for i in range(self.num_local_experts)}

        new_expert_mapping = expert_mapping.copy()
        for i in range(self.config.num_moe_experts):
            remaining_indices = device_full_expert_idx - device_expert_idx[expert_device_mapping[i][0]]
            local_idx = remaining_indices.pop()
            new_expert_mapping[i] = expert_device_mapping[i][0] * self.num_local_experts + local_idx
            device_expert_idx[expert_device_mapping[i][0]].add(local_idx)

        # permuting the experts according to expert_device_mapping
        local_expert_indices_offset = get_rank() % self.ep * self.num_local_experts

        local_expert_device_mapping = np.array(list(expert_device_mapping.values()))[
            local_expert_indices_offset:(local_expert_indices_offset+self.num_local_experts)].flatten()
        new_local_expert_sorted_indices = np.argsort(
            local_expert_device_mapping)

        new_expert_mapping = new_expert_mapping[expert_mapping]
        self.expert_mapping.set_data(Tensor(new_expert_mapping))

        return device_expert_mapping, new_local_expert_sorted_indices
