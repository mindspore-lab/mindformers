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
"""A8 dynamic W8 quantization method."""

from typing import Union

import mindspore
from mindspore import Parameter, nn, ops
from mindspore.common.initializer import initializer
from mindspore.ops.auto_generate import QuantBatchMatmul, DynamicQuantExt, GroupedMatmulV4

from mindformers.parallel_core.inference.tensor_parallel.layers import LinearMethodBase
from mindformers.parallel_core.inference.tensor_parallel.mappings import reduce_from_model_parallel_region
from mindformers.parallel_core.inference.quantization import QuantizationConfig
from mindformers.parallel_core.inference.weights_utils import set_weight_attrs



class A8W8DynamicLinearMethod(LinearMethodBase):
    """Linear method with A8W8 dynamic quantization."""

    def __init__(self, quant_config: QuantizationConfig):
        self.quant_config = quant_config
        self.quant = DynamicQuantExt()
        self.bias_add = ops.Add()

    def create_weights(self,
                       layer: nn.Cell,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int],
                       params_dtype,
                       *weight_args,
                       num_local_experts=None, **extra_weight_attrs) -> Union[Parameter, None]:
        output_size_per_partition = sum(output_partition_sizes)
        self.output_size_per_partition = output_size_per_partition
        self.input_size_per_partition = input_size_per_partition
        self.is_group_mm = num_local_experts is not None

        if self.is_group_mm:
            weight = None
            self.matmul = GroupedMatmulV4()
            if not extra_weight_attrs.get('skip_weight_param_allocation', False):
                shape = (num_local_experts, input_size_per_partition, output_size_per_partition)
                weight = Parameter(initializer('ones', shape, mindspore.int8), requires_grad=False)
                set_weight_attrs(weight, {"input_dim": 1, "output_dim": 2})
                set_weight_attrs(weight, extra_weight_attrs)
                return weight

            w_scale_shape = (num_local_experts, output_size_per_partition)
            w_scale_dtype = mindspore.bfloat16 if params_dtype == mindspore.bfloat16 else mindspore.float32
            w_scale = Parameter(
                initializer('ones', w_scale_shape, w_scale_dtype), name="w_scale", requires_grad=False)
            set_weight_attrs(w_scale, {"output_dim": 1})
            set_weight_attrs(w_scale, extra_weight_attrs)

        else:
            self.matmul = QuantBatchMatmul(transpose_x1=False, transpose_x2=True, dtype=params_dtype)
            weight_shape = (self.output_size_per_partition, self.input_size_per_partition)
            weight = Parameter(initializer('ones', weight_shape, mindspore.int8), requires_grad=False)
            set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
            set_weight_attrs(weight, extra_weight_attrs)

            w_scale_shape = (output_size_per_partition,)
            w_scale_dtype = mindspore.bfloat16 if params_dtype == mindspore.bfloat16 else mindspore.float32
            w_scale = Parameter(
                initializer('ones', w_scale_shape, w_scale_dtype), name="w_scale", requires_grad=False)
            set_weight_attrs(w_scale, {"output_dim": 0})
            set_weight_attrs(w_scale, extra_weight_attrs)

        if layer is not None:
            layer.insert_param_to_cell("weight", weight)
            layer.insert_param_to_cell("w_scale", w_scale)
        return weight


    def apply(self,
              layer: mindspore.nn.Cell,
              x: mindspore.Tensor,
              weight: mindspore.Tensor = None,
              bias: mindspore.Parameter = None,
              group_list=None, **kwargs) -> mindspore.Tensor:
        if weight is None:
            weight = layer.weight
        w_scale = layer.w_scale
        qx, qx_scale = self.quant(x, None)
        qx_scale = qx_scale.reshape(-1)
        output_shape = qx.shape[:-1] + (self.output_size_per_partition,)
        qx = qx.reshape(-1, self.input_size_per_partition)
        if self.is_group_mm:
            out = self.matmul([qx], [weight],
                              None, [w_scale],
                              None,
                              None,
                              None, [qx_scale],
                              group_list,
                              split_item=3,
                              group_type=0,
                              group_list_type=1)[0]
            if hasattr(layer, 'delay_allreduce'):
                if not layer.delay_allreduce and not layer.skip_bias_add:
                    out = reduce_from_model_parallel_region(out, layer.tp_group)
        else:
            out = self.matmul(qx, weight, w_scale, None, None, qx_scale)
        if bias is not None:
            out = self.bias_add(out, bias)
        out = out.reshape(output_shape)
        return out
