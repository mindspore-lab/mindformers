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
"""ModelMixin for infer models."""
import os
from typing import Any, Generator, List, Tuple, Iterable
from safetensors import safe_open
from tqdm.auto import tqdm

from mindspore import Tensor, mutable
from mindspore.nn import Cell
import mindspore.common.dtype as mstype

from mindformers.tools.logger import logger
from mindformers.models.modeling_utils import ModelMixin
from mindformers.parallel_core.inference.parallel_state import is_pipeline_first_stage
from mindformers.version_control import is_310p
from mindformers.parallel_core.inference.tensor_parallel.quantization.base_config import QuantizeMethodBase
from mindformers.parallel_core.inference.transformer.attention import Attention
from mindformers.parallel_core.inference.parallel_state import get_tensor_model_parallel_world_size


class InferModelMixin(ModelMixin):
    """
    A few utilities for `mindspore.nn.Cell`, to be used as a mixin.
    """

    def set_dynamic_inputs(self, **kwargs):
        """ dynamic shape"""
        dynamic_input_ids = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_hidden_states = None if is_pipeline_first_stage() else Tensor(
            shape=[None, None], dtype=self.compute_dtype)
        dynamic_positions = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_context_lens_tensor = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_q_seq_lens = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_attention_mask = Tensor(shape=[None, None], dtype=self.compute_dtype)

        def get_input(fa3_quant=False, fa3_quant_layer=None, need_nz=False):
            if fa3_quant_layer is None:
                fa3_quant_layer = set()
            cache_list = []
            num_layers = len(self.model.decoder.layers)
            for num_layer in range(num_layers):
                kv_cache_dtype = mstype.int8 if fa3_quant and num_layer in fa3_quant_layer else \
                                self.compute_dtype
                if need_nz:
                    cache_list.append(Tensor(shape=[None, None, None], dtype=kv_cache_dtype))
                else:
                    cache_list.append(Tensor(shape=[None, None, None, None], dtype=kv_cache_dtype))
            return mutable(cache_list)

        use_ringmla = getattr(self, 'use_fused_mla', False) and get_tensor_model_parallel_world_size() < 16
        fa3_quant = self.model.quant_config.fa3_quant if self.model.quant_config else False
        fa3_quant_layer = self.model.quant_config.fa3_quant_layer if self.model.quant_config else set()
        if not use_ringmla:
            key_cache = get_input(need_nz=is_310p())
            value_cache = get_input(need_nz=is_310p()) if not self.transformer_config.multi_latent_attention else None
        elif fa3_quant:
            key_cache = get_input(fa3_quant=True, fa3_quant_layer=fa3_quant_layer, need_nz=True)
            value_cache = get_input(need_nz=True)
        else:
            key_cache = get_input(need_nz=is_310p())
            value_cache = get_input(need_nz=is_310p())

        dynamic_attn_padding_idx = None
        dynamic_attn_unpadding_idx = None
        dynamic_ffn_padding_idx = None
        dynamic_ffn_unpadding_idx = None

        tp_group_size = self.model_comm_pgs.tp.size
        dp_group_size = self.model_comm_pgs.dp.size
        ep_group_size = self.model_comm_pgs.moe_ep.size

        # Check whether model needs padding index parameters
        if not (dp_group_size == 1 or (dp_group_size == ep_group_size and tp_group_size == 1)):
            dynamic_attn_padding_idx = Tensor(shape=[None], dtype=mstype.int32)
            dynamic_attn_unpadding_idx = Tensor(shape=[None], dtype=mstype.int32)
            dynamic_ffn_padding_idx = Tensor(shape=[None], dtype=mstype.int32)
            dynamic_ffn_unpadding_idx = Tensor(shape=[None], dtype=mstype.int32)

        # when need padding_idx, add padding parameter into set_input
        self.set_inputs(dynamic_input_ids, dynamic_hidden_states, dynamic_positions, dynamic_batch_valid_length,
                        dynamic_context_lens_tensor, dynamic_q_seq_lens, dynamic_block_tables,
                        dynamic_slot_mapping, dynamic_attention_mask, None,
                        dynamic_attn_padding_idx, dynamic_attn_unpadding_idx,
                        dynamic_ffn_padding_idx, dynamic_ffn_unpadding_idx, key_cache, value_cache)
        logger.info(f"Set dynamic input for {self.__class__.__name__}")

    def add_flags_custom_mcore(self, is_prefill):
        r"""
        Add flag to distinguish fa and pa.

        Args:
            is_prefill: flag to distinguish fa and pa.

        Returns:

        """
        self.add_flags(is_prefill=is_prefill)
        self.model.add_flags(is_prefill=is_prefill)
        self.model.decoder.add_flags(is_prefill=is_prefill)
        for layer in self.model.decoder.layers:
            if self.config.use_flash_attention:
                layer.self_attention.core_attention.add_flags(is_prefill=is_prefill)

    def load_weights(self, weights_path=None, weights: Iterable[Tuple[str, Tensor]] = None):
        r"""
        Load weights.

        Args:
            weights_path: The path of weights.

        """
        if not os.path.isdir(weights_path):
            if not weights:
                raise ValueError(
                    f"Either 'weights_path' or 'weights' is required, "
                    f"but got weights_path={weights_path}, weights={weights}"
                )
            self.model.load_weights(weights)
        else:
            weights_files = [
                os.path.join(weights_path, file)
                for file in os.listdir(weights_path)
                if file.endswith(".safetensors")
            ]

            if not weights_files:
                raise ValueError(f"No .safetensors files found in {weights_path}")

            self.model.load_weights(
                self._safetensors_weights_iterator(weights_files),
                self.generate_mapping()
            )
        self.process_weights_after_loading(self.model)

    def process_weights_after_loading(self, root: Cell, name_prefix: str = "model"):
        """Recursively search for target layers in the network and process_weights_after_loading"""
        if root is None:
            return root
        for name, cell in root.name_cells().items():
            full_cell_name = f"{name_prefix}.{name}"
            quant_method = getattr(cell, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(cell)
                continue
            if isinstance(cell, Attention) and hasattr(cell, "process_weights_after_loading"):
                cell.process_weights_after_loading()
            _ = self.process_weights_after_loading(cell, full_cell_name)
        return root

    def _safetensors_weights_iterator(self, weights_files: List[str]) -> Generator[Tuple[str, Any], None, None]:
        """Iterate over the weights in the model safetensor files."""
        for st_file in tqdm(
                weights_files,
                desc="Loading safetensors checkpoint shards",
        ):
            with safe_open(st_file, framework="np") as f:
                for name in f.keys():  # noqa: SIM118
                    # Return a lightweight PySafeSlice object
                    # that uses file pointer offset internally to read Safetensor
                    # on demand, avoiding memory explosion. Actual data can be obtained through slicing operation
                    # like param[start:end]
                    param = f.get_slice(name)
                    name = self.convert_name(name)
                    yield name, param

    def generate_mapping(self):
        """
        Generate stacked parameter mapping for weight conversion between different model frameworks.

        This method creates a mapping between parameter names in HuggingFace format and
        MindFormers/MCore format, specifically handling cases where multiple parameters
        need to be stacked or merged during the conversion process.

        The mapping rules define how individual weight components (like Q, K, V projections
        or MLP gating/hidden layers) should be mapped to their corresponding stacked
        parameters in the target framework.

        Returns:
            list: A list of tuples, where each tuple contains three elements:
                  - Target parameter name in MCore format
                  - Source parameter name in HuggingFace format
                  - Parameter type identifier for weight loading logic

        Mapping rules cover:
        - Linear projection mappings (Q, K, V, KV projections)
        - MLP layer mappings (gating and hidden layers)
        - Shared expert MLP mappings
        """
        mapping_rules = {
            '.linear_q_down_proj': ('.linear_qkv_down_proj', '.linear_q_down_proj', 'q_down'),
            '.linear_kv_down_proj': ('.linear_qkv_down_proj', '.linear_kv_down_proj', 'kv_down'),
            '.linear_q_up_proj': ('.linear_q_up_proj', '.linear_q_up_proj', 'q_up'),
            '.linear_kv_up_proj': ('.linear_kv_up_proj', '.linear_kv_up_proj', 'kv_up'),
            '.linear_q': ('.linear_qkv', '.linear_q', 'q'),
            '.linear_k': ('.linear_qkv', '.linear_k', 'k'),
            '.linear_v': ('.linear_qkv', '.linear_v', 'v'),
            '.linear_kv': ('.linear_qkv', '.linear_kv', 'kv'),
            '.mlp.gating': ('.mlp.linear_fc1', '.mlp.gating', 'gating'),
            '.mlp.hidden': ('.mlp.linear_fc1', '.mlp.hidden', 'hidden'),
            '.mlp.shared_experts.gating': ('.mlp.shared_experts.linear_fc1', '.mlp.shared_experts.gating', 'gating'),
            '.mlp.shared_experts.hidden': ('.mlp.shared_experts.linear_fc1', '.mlp.shared_experts.hidden', 'hidden')
        }

        stacked_params_mapping = []
        for _, mcore_name in self.weight_mapping:
            for pattern, stacked_param in mapping_rules.items():
                if pattern in mcore_name:
                    stacked_params_mapping.append(stacked_param)
                    break

        return stacked_params_mapping
