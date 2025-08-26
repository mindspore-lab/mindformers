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
"""A8W4 quantization method."""

from typing import Union
import numpy as np

import mindspore
from mindspore import nn, Parameter, ops, mint
from mindspore.common.initializer import initializer
from mindspore.ops.auto_generate import WeightQuantBatchMatmul, DynamicQuantExt, GroupedMatmulV4

from mindformers.parallel_core.inference.weights_utils import set_weight_attrs
from mindformers.parallel_core.inference.transformer.moe.experts import GroupedMLP
from .base_config import QuantizationConfig
from ..layers import LinearMethodBase


class A8W4DynamicLinearMethod(LinearMethodBase):
    """Linear method with A8W4 quantization."""

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
        group_size = self.quant_config.full_config.get("group_size", None)
        if not group_size or group_size < 0:
            raise ValueError(f"group_size should >=0 but group_size is : {group_size}")
        if self.is_group_mm:
            weight = None
            self.matmul = GroupedMatmulV4()
            if not extra_weight_attrs.get('skip_weight_param_allocation', False):
                weight_shape = (num_local_experts, self.input_size_per_partition, self.output_size_per_partition // 2)
                weight = Parameter(initializer('ones', weight_shape, mindspore.qint4x2))
                set_weight_attrs(weight, {"input_dim": 1, "output_dim": 2})
                set_weight_attrs(weight, extra_weight_attrs)
                return weight

            w_scale_shape = (num_local_experts, self.input_size_per_partition // group_size,
                             self.output_size_per_partition)
            w_scale_dtype = mindspore.uint64
            w_scale = Parameter(initializer('ones', w_scale_shape, w_scale_dtype), name="w_scale")
            set_weight_attrs(w_scale, {"input_dim": 1, "output_dim": 2})
            set_weight_attrs(w_scale, extra_weight_attrs)

            gmm_bias = Parameter(initializer("zeros", (w_scale_shape[0], w_scale_shape[-1]), mindspore.float32),
                                 name="gmm_bias")
            set_weight_attrs(gmm_bias, {"output_dim": 1})
            set_weight_attrs(gmm_bias, extra_weight_attrs)

            layer.insert_param_to_cell("gmm_bias", gmm_bias)

        else:
            self.matmul = WeightQuantBatchMatmul(False, True, group_size)
            weight_shape = (self.output_size_per_partition, self.input_size_per_partition)
            weight = Parameter(initializer('ones', weight_shape, mindspore.int8))

            w_scale_shape = (output_size_per_partition,)
            w_scale_dtype = mindspore.bfloat16 if params_dtype == mindspore.bfloat16 else mindspore.float32
            w_scale = Parameter(initializer('ones', w_scale_shape, w_scale_dtype), name="w_scale")

            set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
            set_weight_attrs(w_scale, {"output_dim": 0})

            set_weight_attrs(weight, extra_weight_attrs)
            set_weight_attrs(w_scale, extra_weight_attrs)

        if layer is not None:
            layer.insert_param_to_cell("weight", weight)
            layer.insert_param_to_cell("w_scale", w_scale)

        return weight

    def process_weights_after_loading(self, layer: nn.Cell) -> None:
        """unpack int8 data to int4 in 3dim"""
        if isinstance(layer, GroupedMLP):
            return
        np_data = layer.weight.asnumpy().astype(np.uint8)
        np_data_low = ((np_data & 0x0F) << 4).astype(np.int8) >> 4
        np_data_high = ((np_data >> 4) << 4).astype(np.int8) >> 4

        np_int4_data = np.zeros((np_data.shape[0], np_data.shape[1], np_data.shape[2] * 2),
                                dtype=np.int8)
        np_int4_data[:, :, ::2] = np_data_low
        np_int4_data[:, :, 1::2] = np_data_high
        w_scale = layer.w_scale.asnumpy()
        w_scale_repeat = np.repeat(w_scale, layer.weight.shape[1] // w_scale.shape[1],
                                   axis=1).astype(np.uint32).view(np.float32)
        gmm_bias = 8 * np.sum(
            np_int4_data.astype(np.float32) * w_scale_repeat, axis=1)

        layer.gmm_bias = mindspore.Tensor(gmm_bias, dtype=mindspore.float32)

    def apply(self,
              layer: mindspore.nn.Cell,
              x: mindspore.Tensor,
              weight: mindspore.Tensor = None,
              bias: mindspore.Parameter = None,
              group_list=None,
              **kwargs) -> mindspore.Tensor:
        if weight is None:
            weight = layer.weight
        w_scale = layer.w_scale
        gmm_bias = layer.gmm_bias
        qx, qx_scale = self.quant(x, None)
        qx_scale = qx_scale.reshape(-1)
        output_shape = qx.shape[:-1] + (self.output_size_per_partition,)
        qx = qx.reshape(-1, self.input_size_per_partition)
        if self.is_group_mm:
            w_scale_repeat = ops.cast(w_scale, mindspore.int32)
            w_scale_repeat = ops.cast(w_scale_repeat, mindspore.float32)
            w_scale_repeat = mint.repeat_interleave(w_scale_repeat,
                                                    repeats=weight.shape[1] // w_scale.shape[1],
                                                    dim=1)
            out = self.matmul([qx], [weight],
                              [gmm_bias], [w_scale],
                              None,
                              None,
                              None, [qx_scale],
                              group_list,
                              split_item=3,
                              group_type=0,
                              group_list_type=1)[0]
        else:
            w_scale = ops.cast(w_scale, mindspore.float16)
            qx = ops.cast(qx, mindspore.float16)
            out = self.matmul(qx, weight, w_scale, None, None, None, None)
            out = ops.mul(out, qx_scale.unsqueeze(1))
        if bias is not None:
            out = self.bias_add(out, bias)
        out = out.reshape(output_shape)
        return out
