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
"""Simple GPT model for testing parallel linear layers."""


from functools import partial
import numpy as np
import mindspore as ms
from mindformers.parallel_core.inference.tensor_parallel.layers import (ColumnParallelLinear,
                                                                        RowParallelLinear,
                                                                        MergedColumnParallelLinear,
                                                                        QKVParallelLinear,
                                                                        ReplicatedLinear
                                                                        )
from mindformers.parallel_core.inference.tensor_parallel.grouped_layers import ColumnParallelGroupedLinear
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.parallel_core.inference.quantization.utils import get_quant_config
from mindformers.parallel_core.inference.weights_utils import set_weight_attrs
from mindformers.parallel_core.inference.tensor_parallel.grouped_layers import (
    UnquantizedGroupedLinearMethod
)


def convert_dtype_str_to_ms(dtype_str: str):
    """Convert dtype string to MindSpore dtype."""
    dtype_mapping = {
        'fp32': ms.dtype.float32,
        'fp16': ms.dtype.float16,
        'bf16': ms.dtype.bfloat16,
    }
    mstype = dtype_mapping.get(dtype_str, None)
    if mstype is None:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")
    return mstype


class LinearSpec:
    """Specification for standard linear layers."""

    def __init__(self, linear_type, input_size, output_size, has_bias, compute_dtype, quant_type):
        if isinstance(compute_dtype, str):
            compute_dtype = convert_dtype_str_to_ms(compute_dtype)
        if compute_dtype not in [ms.dtype.float32, ms.dtype.float16, ms.dtype.bfloat16]:
            raise ValueError(f"Unsupported compute_dtype: {compute_dtype}")
        self.linear_type = linear_type
        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = has_bias
        self.skip_bias_add = False
        self.compute_dtype = compute_dtype
        self.transpose_b = True
        self.quant_type = quant_type

    def name(self):
        """Generate a unique name for this layer configuration."""
        return f"{self.linear_type}-has_bias_{self.has_bias}-" \
               f"compute_dtype_{self.compute_dtype}-quant_type_{self.quant_type}"

    def infer_shape(self):
        """Infer output shape. Returns output size."""
        if self.linear_type == "MergedColumnParallelLinear":
            # MergedColumnParallelLinear outputs 2 * output_size (gating + hidden)
            return self.output_size * 2
        return self.output_size


class QKVLinearSpec:
    """Specification for QKV parallel linear layers."""

    def __init__(self, linear_type, hidden_size, head_size, total_num_heads, total_num_kv_heads,
                 has_bias, compute_dtype, quant_type):
        if isinstance(compute_dtype, str):
            compute_dtype = convert_dtype_str_to_ms(compute_dtype)
        if compute_dtype not in [ms.dtype.float32, ms.dtype.float16, ms.dtype.bfloat16]:
            raise ValueError(f"Unsupported compute_dtype: {compute_dtype}")

        self.linear_type = linear_type
        self.input_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        self.output_sizes = [
            total_num_heads * head_size,      # q_proj
            total_num_kv_heads * head_size,   # k_proj
            total_num_kv_heads * head_size,   # v_proj
        ]
        self.has_bias = has_bias
        self.skip_bias_add = False
        self.compute_dtype = compute_dtype
        self.transpose_b = True
        self.quant_type = quant_type

    def name(self):
        """Generate a unique name for this layer configuration."""
        return f"{self.linear_type}-has_bias_{self.has_bias}-" \
               f"compute_dtype_{self.compute_dtype}-quant_type_{self.quant_type}"

    def infer_shape(self):
        """Infer output shape. Returns output size (q + k + v concatenated)."""
        return self.output_size


class GroupLinearSpec:
    """Specification for grouped linear layers (MoE)."""

    def __init__(self, linear_type, num_local_experts, input_size, output_size, quant_type):
        self.linear_type = linear_type
        self.num_local_experts = num_local_experts
        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = None
        self.skip_bias_add = False
        self.transpose_b = True
        self.quant_type = quant_type

    def name(self):
        """Generate a unique name for this layer configuration."""
        return f"{self.linear_type}-has_bias_{self.has_bias}-quant_type_{self.quant_type}"

    def infer_shape(self):
        """Infer output shape. Returns output size."""
        return self.output_size


class ModelSpec:
    """Specification for the entire model."""

    def __init__(self, compute_dtype, param_init_dtype, tensor_parallel, linear_specs):
        self.linear_specs = linear_specs
        self.compute_dtype = compute_dtype
        self.param_init_dtype = param_init_dtype
        self.tensor_parallel = tensor_parallel


class SimplePretrainedConfig(PretrainedConfig):
    """Simple pretrained config for testing."""

    def __init__(self, quantization, pretrained_model_dir):
        super().__init__(
            quantization=quantization,
            pretrained_model_dir=pretrained_model_dir,
        )


class SimpleGPTModel(ms.nn.Cell):
    """A simple GPT model for testing parallel linear operations."""

    def __init__(self, model_spec, comm_pgs, quantization: str, quant_model_dir=None):
        super().__init__()
        self.model_spec = model_spec

        # Setup quantization config
        if quant_model_dir is None:
            quant_config = None
        else:
            quant_config = get_quant_config(SimplePretrainedConfig(quantization, quant_model_dir), [])

        # Setup transformer config
        transformer_config = TransformerConfig(
            tensor_model_parallel_size=model_spec.tensor_parallel,
            compute_dtype=model_spec.compute_dtype,
            params_dtype=model_spec.param_init_dtype,
            num_layers=1,
            num_attention_heads=model_spec.tensor_parallel,
        )

        self.linears = self._build_linears(comm_pgs, model_spec, transformer_config, quant_config)
        self.num_linears = len(self.linears)

    def process_weights_after_loading(self):
        """Process weights after loading - convert format if needed."""
        for cell in self.linears:
            if hasattr(cell, 'quant_method') and cell.quant_method is not None:
                cell.quant_method.process_weights_after_loading(cell)

    @staticmethod
    def _build_linears(comm_pgs, model_spec, transformer_config, quant_config):
        """Build a list of linear layers based on the model specifications."""
        linear_map = {
            "ColumnParallelLinear": partial(ColumnParallelLinear, gather_output=True),
            "ColumnParallelGroupedLinear": partial(ColumnParallelGroupedLinear, gather_output=False),
            "MergedColumnParallelLinear": MergedColumnParallelLinear,
            "QKVParallelLinear": QKVParallelLinear,
            "RowParallelLinear": RowParallelLinear,
            "ReplicatedLinear": ReplicatedLinear,
        }

        linears = []
        for index, linear_spec in enumerate(model_spec.linear_specs):
            linear = SimpleGPTModel._build_single_linear(
                linear_spec, index, linear_map, comm_pgs,
                transformer_config, quant_config
            )
            linears.append(linear)

        return ms.nn.SequentialCell(linears)

    @staticmethod
    def _build_single_linear(linear_spec, index, linear_map, comm_pgs, transformer_config, quant_config):
        """Build a single linear layer based on its specification."""
        linear_type = linear_spec.linear_type
        prefix = f"linears.{index}"

        if linear_type == "QKVParallelLinear":
            return linear_map[linear_type](
                hidden_size=linear_spec.input_size,
                head_size=linear_spec.head_size,
                total_num_heads=linear_spec.total_num_heads,
                total_num_kv_heads=linear_spec.total_num_kv_heads,
                config=transformer_config,
                compute_dtype=linear_spec.compute_dtype,
                transpose_b=linear_spec.transpose_b,
                bias=linear_spec.has_bias,
                tp_group=comm_pgs.tp,
                quant_config=quant_config,
                prefix=prefix
            )

        if linear_type == "MergedColumnParallelLinear":
            return linear_map[linear_type](
                hidden_size=linear_spec.input_size,
                ffn_hidden_size=linear_spec.output_size,
                config=transformer_config,
                bias=linear_spec.has_bias,
                gather_output=True,
                transpose_b=linear_spec.transpose_b,
                compute_dtype=linear_spec.compute_dtype,
                tp_group=comm_pgs.tp,
                quant_config=quant_config,
                prefix=prefix
            )

        if linear_type == "ColumnParallelGroupedLinear":
            # Create weights for grouped linear
            if quant_config is None:
                quant_method = UnquantizedGroupedLinearMethod()
                weight = quant_method.create_weights(
                    layer=None,
                    num_local_experts=linear_spec.num_local_experts,
                    input_size_per_partition=linear_spec.input_size,
                    output_partition_sizes=[linear_spec.output_size],
                    params_dtype=ms.bfloat16
                )
            else:
                quant_method = quant_config.get_quant_method(quant_config, prefix)
                weight = quant_method.create_weights(
                    layer=None,
                    num_local_experts=linear_spec.num_local_experts,
                    input_size_per_partition=linear_spec.input_size,
                    output_partition_sizes=[linear_spec.output_size],
                    params_dtype="bf16"
                )

            linear = linear_map[linear_type](
                num_local_experts=linear_spec.num_local_experts,
                input_size=linear_spec.input_size,
                output_size=linear_spec.output_size,
                config=transformer_config,
                weight=weight,
                bias=linear_spec.has_bias,
                tp_group=comm_pgs.tp,
                quant_config=quant_config,
                prefix=prefix
            )
            set_weight_attrs(weight, {"weight_loader": linear.weight_loader})
            return linear

        # Standard linear layers (ColumnParallelLinear, RowParallelLinear, ReplicatedLinear)
        return linear_map[linear_type](
            input_size=linear_spec.input_size,
            output_size=linear_spec.output_size,
            config=transformer_config,
            skip_bias_add=linear_spec.skip_bias_add,
            compute_dtype=linear_spec.compute_dtype,
            transpose_b=linear_spec.transpose_b,
            bias=linear_spec.has_bias,
            tp_group=comm_pgs.tp,
            quant_config=quant_config,
            prefix=prefix
        )

    def forward(self, x):
        """Forward pass through the model, processing input through all linear layers."""
        outputs = self.construct(x)

        # Process each layer's output into a dictionary
        output_dict = {}
        for index, linear_spec in enumerate(self.model_spec.linear_specs):
            name = f"index_{index}-{linear_spec.name()}"
            output_dict[name] = outputs[index].astype(ms.dtype.float32).asnumpy()

        return output_dict

    def construct(self, x):
        """Forward pass through all layers, returns a list of outputs."""
        outputs = []
        for index in range(self.num_linears):
            linear = self.linears[index]

            # Special handling for grouped linear (MoE)
            if isinstance(linear, ColumnParallelGroupedLinear):
                group_list = np.random.multinomial(
                    x.shape[0],
                    np.ones(linear.num_local_experts) / linear.num_local_experts
                )
                group_list = ms.Tensor(group_list)
                z = linear(x, group_list=group_list)
            else:
                z = linear(x)

            outputs.append(z)

        return outputs
