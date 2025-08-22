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
"""Mindspore quantization configuration."""
from typing import Dict, Any, Optional, List
import re
from collections import defaultdict
import mindspore
from mindformers.parallel_core.inference.tensor_parallel.layers import VocabParallelEmbedding

from . import QuantizationBackends
from .base_config import QuantizationConfig, QuantizeMethodBase
from ..layers import UnquantizedLinearMethod, UnquantizedEmbeddingMethod
from .utils import QUANTIZATION_METHOD_MAPPING, mapping_rules

class MindSporeConfig(QuantizationConfig):
    """Class for Mindspore quantization configs."""

    def __init__(self, full_config: Dict[str, Any]) -> None:
        super().__init__()
        self.full_config = full_config
        self.quantization = full_config["quantization"]
        # osl method need config quantization == golden-stick
        self.is_modelslim = self.quantization != "golden-stick"

    def get_name(self) -> QuantizationBackends:
        return self.quantization

    def get_supported_act_dtypes(self) -> List[str]:
        return [mindspore.dtype.float16, mindspore.dtype.int8]

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_description.json", "quant_model_description.json"]

    def convert_name(self, weight_name):
        "convert name"
        # different param name or weight name converted here
        for hf_name, mcore_name in self.full_config.get("weight_mapping"):
            if hf_name in weight_name:
                weight_name = weight_name.replace(hf_name, mcore_name)

        # FIXME: After osl supports mcore calibration, the following conversion map should be removed.
        weight_name = weight_name.replace('model.tok_embeddings.embedding_weight',
                                          'embedding.word_embeddings.weight')
        weight_name = weight_name.replace('model.norm_out.', 'decoder.final_layernorm.')
        weight_name = weight_name.replace('lm_head.', 'output_layer.')

        weight_name = weight_name.replace('.attention_norm.', '.input_layernorm.')
        weight_name = weight_name.replace('.ffn_norm.', '.pre_mlp_layernorm.')
        weight_name = weight_name.replace('.q2l_proj.', '.linear_q_down_proj.')
        weight_name = weight_name.replace('.lq_norm.', '.q_layernorm.')
        weight_name = weight_name.replace('.l2q_proj.', '.linear_q_up_proj.')
        weight_name = weight_name.replace('.kv2l.', '.linear_kv_down_proj.')
        weight_name = weight_name.replace('.lkv_norm.', '.kv_layernorm.')
        weight_name = weight_name.replace('.lkv2kv.', '.linear_kv_up_proj.')
        weight_name = weight_name.replace('.wo.', '.linear_proj.')

        weight_name = weight_name.replace('.w1.', '.gating.')
        weight_name = weight_name.replace('.w2.', '.linear_fc2.')
        weight_name = weight_name.replace('.w3.', '.hidden.')
        weight_name = weight_name.replace('.routed_experts.router.dense.', '.router.weight.')
        weight_name = weight_name.replace('.routed_experts.router.e_score_correction_bias', '.router.expert_bias')
        weight_name = weight_name.replace('.routed_experts.ffn.', '.experts.')

        weight_name = weight_name.replace('model.layers.', 'decoder.layers.')
        weight_name = weight_name.replace('.attention.', '.self_attention.')
        weight_name = weight_name.replace('.feed_forward.', '.mlp.')

        weight_name = weight_name.replace('.matmul.', '.')
        weight_name = weight_name.replace('.quant_op.', '.')
        weight_name = weight_name.replace('._layer.', '.')

        weight_name = weight_name.replace('.dequant_scale', '.deq_scale')
        weight_name = weight_name.replace('.input_zp', '.input_offset')
        weight_name = weight_name.replace('.weight_scale', '.w_scale')

        return weight_name

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QuantizationConfig":
        return cls(config)

    @classmethod
    def get_min_capability(cls) -> int:
        pass

    def get_quant_method(self, layer: mindspore.nn.Cell, prefix: str) -> Optional[QuantizeMethodBase]:
        self.full_config = {self.convert_name(k): v for k, v in self.full_config.items()}
        get_quant = None
        new_dict = defaultdict(tuple)
        for key, value in mapping_rules.items():
            if len(value) >= 2:
                new_dict[value[0]] += (value[1],)
        for key, value in new_dict.items():
            if key in prefix:
                prefix = prefix.replace(key, value[0])
                break
        for key, value in self.full_config.items():
            if "experts." in prefix:
                key = re.sub(r"experts\.\d+", "experts", key)
            if prefix in key:
                get_quant = value
                break

        if get_quant == "FLOAT":
            if "embedding" in prefix and isinstance(layer, VocabParallelEmbedding):
                quant_method = UnquantizedEmbeddingMethod()
            else:
                quant_method = UnquantizedLinearMethod()
            return quant_method
        quant_method = QUANTIZATION_METHOD_MAPPING[get_quant]
        if quant_method is None:
            raise ValueError(f"Unknown quantization method: {quant_method}")
        return quant_method(self)
