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
"""Expert GrouedMLP."""
import mindspore.ops.functional as F
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.common.initializer import initializer

from mindformers.parallel_core.inference.utils import get_tp_world_size
from mindformers.parallel_core.inference.utils import divide
from mindformers.parallel_core.inference.transformer.activation import get_act_func
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.tensor_parallel.random import (TENSOR_PARALLEL_GENERATOR,
                                                                        get_rng_tracer)
from mindformers.parallel_core.inference.tensor_parallel.mappings import (ReduceFromModelParallelRegion,
                                                                          GatherFromModelParallelRegion)

__all__ = [
    "GroupedMLP",
]


class GroupedMLP(nn.Cell):
    """An implementation of the Experts layer.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(self, num_experts: int, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.has_bias = self.config.add_bias_linear
        self.gated_linear_unit = self.config.gated_linear_unit
        self.activation_type = self.config.hidden_act
        self.params_dtype = self.config.params_dtype
        self.compute_dtype = self.config.compute_dtype
        self.moe_delay_allreduce = True
        self.skip_bias_add = True


        ffn_hidden_size = self.config.moe_shared_expert_intermediate_size
        self.tp_group_size = get_tp_world_size()
        self.ffn_hidden_size_per_partition = divide(ffn_hidden_size, self.tp_group_size)
        if self.gated_linear_unit:
            mapping_ffn_hidden_size = ffn_hidden_size * 2
        else:
            mapping_ffn_hidden_size = ffn_hidden_size

        # linear fc1
        self.linear_fc1_input_size = self.config.hidden_size
        self.linzer_fc1_output_size = mapping_ffn_hidden_size
        self.linear_fc1_output_size_per_partition = divide(self.linzer_fc1_output_size, self.tp_group_size)
        linear_fc1_weight_shape = (self.num_experts,) + (self.linear_fc1_input_size,
                                                         self.linear_fc1_output_size_per_partition)

        if self.activation_type is not None:
            self.activation_func = get_act_func(self.activation_type)
        else:
            self.activation_func = None
        # linear fc2
        self.linear_fc2_input_size = self.config.moe_shared_expert_intermediate_size
        self.linzer_fc2_output_size = self.config.hidden_size
        self.linear_fc2_output_size_per_partition = divide(self.linzer_fc2_output_size, self.tp_group_size)
        self.linear_fc2_input_size_per_partition = divide(self.linear_fc2_input_size, self.tp_group_size)
        linear_fc2_weight_shape = (self.num_experts,) + (self.linear_fc2_input_size,
                                                         self.linear_fc2_output_size_per_partition)

        self.cast = ops.Cast()
        self.matmul = ops.auto_generate.GroupedMatmulV4()
        self.reduce_from_mp_region = ReduceFromModelParallelRegion()
        self.gather_from_mp_region = GatherFromModelParallelRegion()

        with get_rng_tracer().rng_fork(TENSOR_PARALLEL_GENERATOR):
            self.weight1 = Parameter(initializer("normal", linear_fc1_weight_shape, self.params_dtype), name="weight")
            self.weight2 = Parameter(initializer("normal", linear_fc2_weight_shape, self.params_dtype), name="weight")

        if self.has_bias:
            self.linear_fc1_bias = Parameter(
                initializer(
                    "zeros", (self.linear_fc1_output_size_per_partition), self.params_dtype
                ),
                name="bias",
            )
            bias_shape = (1, self.num_experts, 1) + (self.linzer_fc2_output_size,)
            self.linear_fc2_bias = Parameter(
                initializer(
                    "zeros", bias_shape, self.params_dtype
                ),
                name="bias",
            )

    def _fc1_group_gemm(self, input_parallel: Tensor, weight=None, group_list=None,
                        input_size=None, output_size=None, bias=None):
        """Using grouped matmul to do compute."""
        origin_dtype = F.dtype(input_parallel)
        weight = self.cast(weight, self.compute_dtype)
        input_parallel = self.cast(input_parallel, self.compute_dtype)
        output_size_per_partition = divide(output_size, self.tp_group_size)
        output_shape = input_parallel.shape[:-1] + (output_size_per_partition,)
        input_parallel = mint.reshape(input_parallel, (-1, input_size))
        output_parallel = self.matmul([input_parallel], [weight], None, None, None, None, None, None,
                                      group_list, split_item=3, group_type=0, group_list_type=1)[0]
        if self.has_bias:
            output_parallel = mint.add(
                output_parallel, self.cast(bias, self.compute_dtype)
            )
        output_parallel = self.cast(output_parallel, origin_dtype)
        output_parallel = mint.reshape(output_parallel, output_shape)
        output_parallel = self.gather_from_mp_region(output_parallel)

        return output_parallel

    def _fc2_group_gemm(self, input_parallel: Tensor, weight=None, group_list=None,
                        input_size=None, output_size=None, bias=None):
        """Using grouped matmul to do compute."""
        origin_dtype = F.dtype(input_parallel)
        weight = self.cast(weight, self.compute_dtype)
        input_parallel = self.cast(input_parallel, self.compute_dtype)
        output_shape = input_parallel.shape[:-1] + (output_size,)
        input_size_per_partition = divide(input_size, self.tp_group_size)
        input_parallel = mint.reshape(input_parallel, (-1, input_size_per_partition))
        output_parallel = self.matmul([input_parallel], [weight], None, None, None, None, None, None,
                                      group_list, split_item=3, group_type=0, group_list_type=1)[0]
        if self.moe_delay_allreduce:
            output = output_parallel
        else:
            output = self.reduce_from_mp_region(output_parallel)

        if self.has_bias and not self.skip_bias_add:
            output = mint.add(output, self.cast(bias, self.compute_dtype))

        output = self.cast(output, origin_dtype)
        output = mint.reshape(output, output_shape)

        return output_parallel

    def construct(self, hidden_states, group_list=None):
        """Forward process of GroupedMLP"""
        # [T, H] -> [T, ffn_H]
        intermediate_parallel = self._fc1_group_gemm(hidden_states, self.weight1, group_list,
                                                     self.linear_fc1_input_size, self.linzer_fc1_output_size)

        if self.gated_linear_unit:
            gate, hidden = mint.split(intermediate_parallel,
                                      (self.ffn_hidden_size_per_partition,
                                       self.ffn_hidden_size_per_partition), -1)
            gate = self.activation_func(gate) if self.activation_type else gate
            intermediate_parallel = mint.mul(hidden, gate)
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel) if self.activation_type else intermediate_parallel

        # [T, ffn_H] -> [T, H]
        output = self._fc2_group_gemm(intermediate_parallel, self.weight2, group_list,
                                      self.linear_fc2_input_size, self.linzer_fc2_output_size)
        return output

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (1, self.tensor_parallel_group_size, 1)

        state_dict = {}
        state_dict[self.weight.name] = {'shape': self.weight.shape,
                                        'shard': w_shard}
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (1,)}
        return state_dict
