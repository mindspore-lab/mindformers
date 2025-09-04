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

from mindformers.parallel_core.inference.tensor_parallel.layers import (VocabParallelEmbedding,
                                                                        UnquantizedLinearMethod,
                                                                        UnquantizedEmbeddingMethod)
from mindformers.parallel_core.inference.quantization import QuantizationConfig, QuantizationBackends
from mindformers.parallel_core.inference.quantization.base_config import QuantizeMethodBase
from mindformers.parallel_core.inference.quantization.golden_stick.a8w8 import A8W8LinearMethod
from mindformers.parallel_core.inference.quantization.golden_stick.a8dynw8 import A8W8DynamicLinearMethod
from mindformers.parallel_core.inference.quantization.golden_stick.a8dynw4 import A8W4DynamicLinearMethod

QUANTIZATION_METHOD_MAPPING = {
    "W8A8": A8W8LinearMethod,
    "W8A8_DYNAMIC": A8W8DynamicLinearMethod,
    "W4A8_DYNAMIC": A8W4DynamicLinearMethod
}

mapping_rules = {
    '.linear_q_down_proj': ('.linear_qkv_down_proj', '.linear_q_down_proj', 'q_down'),
    '.linear_kv_down_proj': ('.linear_qkv_down_proj', '.linear_kv_down_proj', 'kv_down'),
    '.linear_q': ('.linear_qkv', '.linear_q', 'q'),
    '.linear_k': ('.linear_qkv', '.linear_k', 'k'),
    '.linear_v': ('.linear_qkv', '.linear_v', 'v'),
    '.linear_kv': ('.linear_qkv', '.linear_kv', 'kv'),
    '.mlp.gating': ('.mlp.linear_fc1', '.mlp.gating', 'gating'),
    '.mlp.hidden': ('.mlp.linear_fc1', '.mlp.hidden', 'hidden'),
    '.mlp.shared_experts.gating': ('.mlp.shared_experts.linear_fc1', '.mlp.shared_experts.gating', 'gating'),
    '.mlp.shared_experts.hidden': ('.mlp.shared_experts.linear_fc1', '.mlp.shared_experts.hidden', 'hidden'),
    '.mlp.experts.gating': ('.mlp.experts.linear_fc1', '.mlp.experts.gating', 'gating'),
    '.mlp.experts.hidden': ('.mlp.experts.linear_fc1', '.mlp.experts.hidden', 'hidden')
}


class GoldenStickConfig(QuantizationConfig):
    """Class for Mindspore quantization configs."""

    def __init__(self, full_config: Dict[str, Any]) -> None:
        super().__init__()
        self.full_config = full_config
        self.quantization = full_config.get("quantization", None)
        # osl method need config quantization == golden-stick
        self.is_modelslim = self.quantization != "golden-stick"
        self.fa3_quant = full_config.get("fa_quant_type", None) == "FAQuant"
        self.fa3_quant_layer = self.get_fa3_quant_layer() if self.fa3_quant else set()

    def get_name(self) -> QuantizationBackends:
        return self.quantization

    def get_supported_act_dtypes(self) -> List[str]:
        return [mindspore.dtype.float16, mindspore.dtype.int8]

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_description.json", "quant_model_description.json"]

    def convert_name(self, weight_name):
        """
        If the layer prefix in the quantization strategy json file is different from the network,
        transform it to network's corresponding layer prefix.
        """

        for hf_name, mcore_name in self.full_config.get("weight_mapping"):
            if hf_name in weight_name:
                weight_name = weight_name.replace(hf_name, mcore_name)

        # After osl supports mcore calibration, the following conversion map should be removed.
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

    def get_fa3_quant_layer(self) -> set[int]:
        fa3_quant_layer = set()
        fa3_quant_pattern = r'^model\.layers\.(\d+)\.self_attn\.fa_q\.scale$'
        for key, _ in self.full_config.items():
            match = re.fullmatch(fa3_quant_pattern, key)
            if match:
                fa3_quant_layer.add(int(match.group(1)))
        return fa3_quant_layer

    def get_quant_method(self, layer: mindspore.nn.Cell, prefix: str) -> Optional[QuantizeMethodBase]:
        """
        This method is used to obtain the quant_method of each layer of the network by matching
        the corresponding quantization strategy according the layer prefix from a quantization
        strategy json file where a mapping between the layer prefix and the corresponding
        quantization strategy is stored in, then mapping the quantization strategy
        to the corresponding quant_method.
        """
        self.full_config = {self.convert_name(k): v for k, v in self.full_config.items()}
        quant_strategy = None
        mapping_dict = defaultdict(tuple)

        # for the fusion layer prefix of the network, if there is no corresponding layer prefix in
        # the quantization strategy json, it need to be mapped to a sub-layer of fusion layer
        # according the mapping_rules to index quantization strategy of this fusion layer.
        for key, value in mapping_rules.items():
            if len(value) >= 2:
                mapping_dict[value[0]] += (value[1],)
        for key, value in mapping_dict.items():
            if key in prefix:
                prefix = prefix.replace(key, value[0])
                break
        # for multiple experts, every expert has the same quantization strategy, so the fuzzy
        # matching ignores the expert id in the expert layer prefix of the quantization
        # strategy json.
        for key, value in self.full_config.items():
            if "experts." in prefix:
                key = re.sub(r"experts\.\d+", "experts", key)
            if prefix in key:
                quant_strategy = value
                break
        # mapping the quant_strategy to the corresponding quant_method
        if not quant_strategy:
            raise ValueError(f"for this layer: {prefix} No corresponding matching quantization strategy"
                             "found in the quantization strategy json file")
        if quant_strategy == "FLOAT":
            if "embedding" in prefix and isinstance(layer, VocabParallelEmbedding):
                quant_method = UnquantizedEmbeddingMethod()
            else:
                quant_method = UnquantizedLinearMethod()
            return quant_method
        quant_method = QUANTIZATION_METHOD_MAPPING[quant_strategy]
        if quant_method is None:
            raise ValueError(f"Unknown quantization method: {quant_method}")
        return quant_method(self)
