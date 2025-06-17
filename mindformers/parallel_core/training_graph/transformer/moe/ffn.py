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
"""FFN layer for MoE."""
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from mindspore.context import ParallelMode
from mindspore.ops.auto_generate import Cast, GroupedMatmul, Reshape, Swiglu
from mindspore.ops.operations import Morph
from mindspore.parallel.shard import Layout
from mindspore.parallel._utils import _get_parallel_mode

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher


def func_infer_dtype(*args):
    """infer_dtype for Morph."""
    return args[0]


def func_infer_shape(*args):
    """infer_shape for Morph."""
    return args[0]


class FFNGroupedGEMM(nn.Cell):
    """
    Initializes a Feed-Forward Network (FFN) cell, which is a fundamental building block in many
    neural network architectures, especially within transformer models.

    Args:
        config (TransformerConfig): Configuration object for the FFN module.
    """

    def __init__(self, config: TransformerConfig):
        super(FFNGroupedGEMM, self).__init__()
        self.config = config
        self.num_local_experts = config.num_moe_experts

        if config.add_bias_linear:
            raise NotImplementedError(
                "bias not supported in FFNGroupedGEMM yet, please set: "
                "model_config: \n"
                "    add_bias_linear: True' \n"
                "in yaml configuration.")

        # activation_func is not in the supported set for now, Swiglu by default.

        # config.gated_linear_unit=True for MOE module by default.
        # Note: The gated_linear_unit config switch will be supported later.
        fc1_output_size = 2 * self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        self.intermediate_size = self.config.moe_ffn_hidden_size
        self.hidden_size = config.hidden_size

        self.compute_dtype = config.compute_dtype
        self.param_init_type = config.params_dtype
        self.init_method = config.init_method
        self.ep = config.expert_model_parallel_size
        self.dp = config.data_parallel_size * config.tensor_model_parallel_size

        # parameters
        self.weight1 = Parameter(
            self.init_method([self.hidden_size, fc1_output_size]),
            name='w1')
        self.weight2 = Parameter(
            self.init_method([fc2_input_size, self.hidden_size]),
            name='w2')

        # init token dispatcher
        self.token_dispatcher = MoEAlltoAllTokenDispatcher(config)

        # init morphed layer
        self.morphed_forward = Morph(self.forward_func, func_infer_shape, func_infer_dtype).add_prim_attr(
            "self_define_shard", True)

        self.cast_op = Cast()
        self.reshape = Reshape()

        if _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard(config)

    def construct(self, tokens, probs, routing_map):
        """
        Construct function of FFN.
        This module integrates the computations of both the token_dispatcher and experts into the Morph operator.
        The detailed implementation can be found in the functions encapsulated within the Morph operator.
        """
        dtype = tokens.dtype
        w1 = self.cast_op(self.weight1, dtype)
        w2 = self.cast_op(self.weight2, dtype)
        # reshape w1 and w2 to [E, h, H] and [E, H, h]
        w1 = self.reshape(w1, (self.num_local_experts, self.hidden_size, 2 * self.intermediate_size))
        w2 = self.reshape(w2, (self.num_local_experts, self.intermediate_size, self.hidden_size))
        output = self.morphed_forward(tokens, probs, routing_map, w1, w2)
        return output

    def forward_func(self, tokens, probs, routing_map, w1, w2):
        """Morphed forward."""
        (dispatched_input, tokens_per_expert, unsort_map, outer_unsort_map,
         input_splits, output_splits, original_shape, unsort_pad_map) = \
            self.token_dispatcher.token_permutation(tokens, probs, routing_map)

        expert_output = self.experts_forward(dispatched_input, tokens_per_expert, w1, w2)

        output = self.token_dispatcher.token_unpermutation(
            expert_output, probs, unsort_map, outer_unsort_map,
            input_splits, output_splits, original_shape, unsort_pad_map)
        return output

    def experts_forward(self, permuted_local_hidden_states, tokens_per_expert, w1, w2):
        """Forward step of the GroupedMLP.
        Matches moe.experts.GroupedMLP's computation precision of Megatron v0.12.0
        under certain conditions.
        """
        # Only supported  permuted_local_hidden_states.nelement() != 0
        # MindFormers use use_pad_tokens control when permuted_local_hidden_states.nelement() == 0
        # in moe.token_dispatcher.MoEAlltoAllTokenDispatcher._process_pad_tokens
        permuted_local_hidden_states = permuted_local_hidden_states.reshape((-1, self.hidden_size))
        fc1_output = GroupedMatmul(split_item=3, group_type=0)(
            [permuted_local_hidden_states], [w1], None, None, None, None, None, tokens_per_expert)[0]
        intermediate_parallel = Swiglu()(fc1_output, -1).reshape((-1, w2.shape[1]))
        fc2_output = GroupedMatmul(split_item=3, group_type=0)(
            [intermediate_parallel], [w2], None, None, None, None, None, tokens_per_expert)[0]
        fc2_output = fc2_output.reshape((1, -1, self.hidden_size))
        return fc2_output

    def shard(self, config: TransformerConfig):
        """
        Handles the sharding configuration for the model's parallelism settings.
        """
        dp = config.data_parallel_size * config.tensor_model_parallel_size * config.context_parallel_size
        ep = config.expert_model_parallel_size
        if dp % ep != 0:
            raise ValueError(
                f"The value of expert_model_parallel_size must be divisible by "
                "data_parallel_size * tensor_model_parallel_size, where"
                f"data_parallel_size: {config.data_parallel_size}, "
                f"tensor_model_parallel_size: {config.tensor_model_parallel_size}, "
                f"expert_model_parallel_size: {config.expert_model_parallel_size}.")
        outer_dp = dp // ep
        inner_dp = ep

        layout = Layout((outer_dp, inner_dp, 1, 1, 1), ("outer_dp", "inner_dp", "sp", "mp0", "mp1"))
        self.morphed_forward.shard(
            in_strategy=(
                layout(("outer_dp", "inner_dp"), "sp", "mp0"),  # tokens       [B, S, h]
                layout(("outer_dp", "inner_dp"), "sp", "mp0"),  # probs        [B, S, k]
                layout(("outer_dp", "inner_dp"), "sp", "mp0"),  # routing_map  [B, S, k]
                layout("inner_dp", "mp0", "mp1"),  # w1  [E, h, H]
                layout("inner_dp", "mp1", "mp0"),  # w2  [E, H, h]
            ),
            out_strategy=(
                layout(("outer_dp", "inner_dp"), "sp", "mp0"),  # output       [B, S, h]
            )
        )
