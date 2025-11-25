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
import numpy as np

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore.context import ParallelMode
from mindspore.ops.operations import Morph
from mindspore.ops.auto_generate import AddExt, Reshape, Shape, Transpose
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.parallel_core.training_graph.device_matrix import layout, layout_moe
from mindformers.parallel_core.training_graph.transformer.moe.router import TopKRouter
from mindformers.parallel_core.training_graph.transformer.moe.utils import get_dp_mod_ep_group_name
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

        if self.config.print_expert_load:
            self.expert_load_history = Parameter(
                Tensor(np.zeros(self.config.num_moe_experts), ms.float32), requires_grad=False)
            self.expert_load_history_cnt = Parameter(
                Tensor(0, dtype=ms.int32), requires_grad=False)

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
        self.seq_length = config.seq_length
        self.use_seq_parallel = config.sequence_parallel
        self.dp = config.data_parallel_size * config.tensor_model_parallel_size * config.context_parallel_size
        self.tp = config.tensor_model_parallel_size
        self.cp = config.context_parallel_size

        if self.config.print_expert_load:
            self.ep = config.expert_model_parallel_size
            self.expert_num = config.num_moe_experts
            self.num_local_experts = self.expert_num // self.ep
            self.dp_modulo_ep_group = get_dp_mod_ep_group_name(self.dp, self.ep)

        # ops
        self.add = AddExt()
        if self.shared_expert_overlap:
            self.add.add_prim_attr("parallel_branch", 1)
        self.reshape = Reshape()
        # Reshape and permute input tensor from (B,S,H) to (dp, N, H)
        self.local_reshape_permute = Morph(self.permute_reshape_fn,
                                           self.permute_reshape_infer_shape,
                                           self.permute_reshape_infer_dtype
                                           ).add_prim_attr("self_define_shard", True)
        # Reshape and unpermute tensor back from (dp, N, H) to (B,S,H)
        self.local_reshape_unpermute = Morph(self.unpermute_reshape_fn,
                                             self.unpermute_reshape_infer_shape,
                                             self.unpermute_reshape_infer_dtype
                                             ).add_prim_attr("self_define_shard", True)

        self.shape = Shape()
        self.add_loss = AddExt()
        self.transpose = Transpose()
        self.transpose2 = Transpose()

        #check_rules
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

    def permute_reshape_infer_shape(self, *args):
        origin_shape = args[0]
        return (self.dp, origin_shape[0] * origin_shape[1] // self.dp, origin_shape[-1])

    def unpermute_reshape_infer_shape(self, *args):
        permute_shape = args[0]
        return (permute_shape[0] * permute_shape[1] // self.seq_length, self.seq_length, permute_shape[-1])

    def permute_reshape_infer_dtype(self, *args):
        return args[0]

    def unpermute_reshape_infer_dtype(self, *args):
        return args[0]

    def permute_reshape_fn(self, x, shp):
        return Reshape()(x, shp)

    def unpermute_reshape_fn(self, x, shp):
        return Reshape()(x, shp)

    # pylint: disable=W0237
    def construct(self, hidden_states, extra_loss=0., seq_chunk=None):
        """Construct function of the MoELayer."""
        x = self.transpose(hidden_states, (1, 0, 2))
        origin_shape = self.shape(x)
        # The shape change is: (dp, N, h) <-- (B*S, h)
        # Do local reshape to avoid unnecessary redistribution, cause the router and ffn only related to tokens.
        x_reshaped = self.local_reshape_permute(x, (1, -1, self.hidden_size))

        # 1. router
        expert_index, router_coeff, router_aux_loss = self.router(x_reshaped)

        # 2. permute + experts + unpermute
        experts_output = self.experts(x_reshaped, router_coeff, expert_index)

        # Do local reshape to avoid not unnecessary redistribution, cause the router and ffn only related to tokens.
        experts_output = self.local_reshape_unpermute(experts_output,
                                                      (-1, self.seq_length // (self.tp * self.cp), origin_shape[-1]))
        experts_output = self.transpose2(experts_output, (1, 0, 2))

        # 3. shared experts
        if self.use_shared_expert:
            shared_experts_output, _ = self.shared_experts(hidden_states)
            experts_output = self.add(experts_output, shared_experts_output)

        extra_loss = self.add_loss(extra_loss, router_aux_loss)
        return experts_output, None, extra_loss

    def shard(self, config: TransformerConfig):
        """Set parallel strategy."""
        self.transpose.shard((layout("cp_tp", "dp", "None"),))
        self.local_reshape_permute.shard(in_strategy=(layout("dp", "cp_tp", "None"),),
                                         out_strategy=(layout_moe("dp_cp", "None", "None"),))
        self.local_reshape_unpermute.shard(in_strategy=(layout_moe("dp_cp", "None", "None"),),
                                           out_strategy=(layout("dp", "cp_tp", "None"),))
        if self.use_seq_parallel:
            self.transpose2.shard((layout("dp", "cp_tp", "None"),))
            self.add.shard((layout("cp_tp", "dp", "None"), layout("cp_tp", "dp", "None")))
        else:
            self.transpose2.shard((layout("dp", "cp", "None"),))
            self.add.shard((layout("cp", "dp", "None"), layout("cp", "dp", "None")))

    def update_expert_load_history(self, num_tokens_per_expert):
        """
        Update expert load history based on token distribution.

        Args:
            num_tokens_per_expert: Array containing number of tokens assigned to each expert.
        """
        # update expert load
        expert_load_history_cnt_new = self.expert_load_history_cnt + 1
        expert_load_new = ops.cast((self.expert_load_history * self.expert_load_history_cnt
                                    + num_tokens_per_expert) / (expert_load_history_cnt_new), ms.float32)

        expert_load_gathered = self.gather_expert_load_data_parallel(expert_load_new)
        self.expert_load_history.set_data(expert_load_gathered)
        if self.config.print_expert_load:
            expert_load_history_cnt_new = ops.minimum(
                expert_load_history_cnt_new, Tensor(100, ms.int32))

        self.expert_load_history_cnt.set_data(expert_load_history_cnt_new)

    def gather_expert_load_data_parallel(self, num_tokens_per_expert):
        dp_modulo_ep_group_size = self.dp // self.ep
        if dp_modulo_ep_group_size > 1:
            num_tokens_per_expert_overall = ops.AllReduce(
                group=self.dp_modulo_ep_group)(num_tokens_per_expert)
            num_tokens_per_expert_overall = num_tokens_per_expert_overall // dp_modulo_ep_group_size
        else:
            num_tokens_per_expert_overall = num_tokens_per_expert.clone()
        return num_tokens_per_expert_overall
