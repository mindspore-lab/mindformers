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
"""a model designed for test."""


from functools import partial
import numpy as np
import mindspore as ms
from mindformers.parallel_core.inference.tensor_parallel.layers import (ColumnParallelLinear,
                                                                        RowParallelLinear,
                                                                        MergedColumnParallelLinear,
                                                                        QKVParallelLinear,
                                                                        ReplicatedLinear)
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.parallel_core.inference.quantization.utils import get_quant_config


class LinearSpec:
    """Specification for linear layers in the model."""

    def __init__(self, linear_type, input_size, output_size, has_bias, compute_dtype, quant_type):
        if isinstance(compute_dtype, str):
            compute_dtype = self.convert_pt_dtype_to_ms(compute_dtype)
        if compute_dtype not in [ms.dtype.float32, ms.dtype.float16, ms.dtype.bfloat16]:
            raise ValueError(f"Unsupported compute_dtype: {compute_dtype}")
        self.linear_type = linear_type
        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = has_bias
        self.skip_bias_add = False
        self.compute_dtype = compute_dtype
        self.transpose_b=True
        self.quant_type = quant_type

    def name(self):
        return f"{self.linear_type}-has_bias_{self.has_bias}-" \
               f"compute_dtype_{self.compute_dtype}-quant_type_{self.quant_type}"

    @staticmethod
    def convert_pt_dtype_to_ms(pt_dtype: str):
        """Convert PyTorch dtype to MindSpore dtype."""
        dtype_mapping = {
            'fp32': ms.dtype.float32,
            'fp16': ms.dtype.float16,
            'bf16': ms.dtype.bfloat16,
        }
        mstype = dtype_mapping.get(pt_dtype, None)
        if mstype is None:
            raise ValueError(f"Unsupported pytorch dtype: {pt_dtype}")
        return mstype


class ModelSpec:
    def __init__(self, compute_dtype, param_init_dtype, tensor_parallel, linear_specs):
        self.linear_specs = linear_specs
        self.compute_dtype = compute_dtype
        self.param_init_dtype = param_init_dtype
        self.tensor_parallel = tensor_parallel


class TestPretrainedConfig(PretrainedConfig):
    def __init__(self, quantization, pretrained_model_dir):
        super().__init__(
            quantization=quantization,
            pretrained_model_dir=pretrained_model_dir,
        )


class GPTModelForTest(ms.nn.Cell):
    """A model designed for testing parallel linear operations."""

    def __init__(self, model_spec, comm_pgs, quantization: str, quant_model_dir=None):
        super().__init__()
        self.model_spec = model_spec
        if quant_model_dir is None:
            quant_config = None
        else:
            quant_config = get_quant_config(TestPretrainedConfig(quantization, quant_model_dir), [])
        transformer_config = TransformerConfig(
            tensor_model_parallel_size=model_spec.tensor_parallel,
            compute_dtype=model_spec.compute_dtype,
            params_dtype=model_spec.param_init_dtype,
            num_layers=1,
            num_attention_heads=model_spec.tensor_parallel,
        )
        self.linears = GPTModelForTest._build_linears(comm_pgs, model_spec, transformer_config, quant_config)
        self.num_linears = len(self.linears)

    @staticmethod
    def _build_linears(comm_pgs, model_spec, transformer_config, quant_config):
        """Build a list of linear layers based on the model specifications."""
        linear_map = {
            "ColumnParallelLinear": partial(ColumnParallelLinear, gather_output=True),
            "MergedColumnParallelLinear": MergedColumnParallelLinear,
            "QKVParallelLinear": QKVParallelLinear,
            "RowParallelLinear": RowParallelLinear,
            "ReplicatedLinear": ReplicatedLinear,
        }
        linears = []
        for index, linear_spec in enumerate(model_spec.linear_specs):
            linear = linear_map[linear_spec.linear_type](
                input_size=linear_spec.input_size,
                output_size=linear_spec.output_size,
                config=transformer_config,
                skip_bias_add=linear_spec.skip_bias_add,
                compute_dtype=linear_spec.compute_dtype,
                transpose_b=linear_spec.transpose_b,
                bias=linear_spec.has_bias,
                tp_group=comm_pgs.tp,
                quant_config=quant_config,
                prefix=f"linears.{index}"
            )
            linears.append(linear)
        return ms.nn.SequentialCell(linears)

    def forward(self, x):
        """Forward pass through the model, processing input through all linear layers."""
        output = self.construct(x).astype(ms.dtype.float32).asnumpy()
        bs = output.shape[0]
        if bs != self.num_linears:
            raise ValueError(f"outputs size must be equal to the number of linears: {bs} != {self.num_linears}")
        outputs = np.split(output, bs, axis=0)
        output_dict = {}
        for index, linear_spec in enumerate(self.model_spec.linear_specs):
            name = f"index_{index}-{linear_spec.name()}"
            output_dict[name] = outputs[index].squeeze(axis=0)
        return output_dict

    def construct(self, x):
        y = ms.ops.zeros_like(x)
        y = y.expand_dims(axis=0)
        for index in range(self.num_linears):
            linear = self.linears[index]
            z = linear(x).expand_dims(axis=0)
            y = ms.ops.concat((y, z))
        return y[1:,::]
