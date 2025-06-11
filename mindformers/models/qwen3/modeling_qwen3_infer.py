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
"""Qwen3 models' APIs."""
__all__ = ['InferenceQwen3ForCausalLM']

from typing import Dict
import gc
import os
from safetensors import safe_open
from tqdm import tqdm

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, mutable
from mindspore.communication import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication._comm_helper import _is_initialized

from mindformers.models.utils import jit
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.logger import logger
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.transformer_config_utils import convert_to_transformer_config
from mindformers.models.qwen3.utils import Qwen3PreTrainedModel
from mindformers.parallel_core.inference.parallel_state import (
    get_group_info,
    initialize_model_parallel
)
from mindformers.parallel_core.inference.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class InferenceQwen3ForCausalLM(Qwen3PreTrainedModel):
    r"""
    Provide qwen3 model infer through network.

    Args:
        config (Qwen3Config): The config of qwen3 model.

    Returns:
        output: Tensor, the output of qwen3 decoder layer

    """

    def __init__(self, config):
        super().__init__(config, auto_prefix=False)
        if get_group_info('tp').group is None and _is_initialized():
            initialize_model_parallel(get_group_size(), order='tp')
        self.config = config
        config: TransformerConfig = convert_to_transformer_config(self.config)
        self.pad_token_id = self.config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.compute_dtype = config.compute_dtype

        self.is_prefill = True
        if isinstance(self.config.parallel_decoding_params, Dict):
            self.plugin_type = self.config.parallel_decoding_params.get("plugin_type")
        else:
            self.plugin_type = None
        self.model = GPTModel(config=config,
                              transformer_layer_spec=get_gpt_layer_local_spec(
                                  normalization=config.normalization,
                                  use_flash_attention=self.config.use_flash_attention,
                                  qk_layernorm=True,
                              ),
                              vocab_size=self.vocab_size,
                              max_sequence_length=self.max_position_embeddings,
                              position_embedding_type=config.position_embedding_type,
                              rotary_base=self.config.rope_theta,
                              tie_word_embeddings=self.config.tie_word_embeddings,
                              post_process=self.config.post_process)

    def set_dynamic_inputs(self, **kwargs):
        """ dynamic shape"""
        dynamic_input_ids = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_positions = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_context_lens_tensor = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_q_seq_lens = Tensor(shape=[None], dtype=mstype.int32)

        dynamic_attention_mask = Tensor(shape=[None, None], dtype=self.compute_dtype)

        def get_input():
            cache_list = []
            for _ in range(self.config.num_hidden_layers):
                cache_list.append(Tensor(shape=[None, None, None, None], dtype=self.config.compute_dtype))
            return mutable(cache_list)
        key_cache = get_input()
        value_cache = get_input()

        self.set_inputs(dynamic_input_ids, dynamic_positions, dynamic_batch_valid_length,
                        dynamic_context_lens_tensor, dynamic_q_seq_lens, dynamic_block_tables,
                        dynamic_slot_mapping, dynamic_attention_mask, None, key_cache, value_cache)
        logger.info("Set dynamic input for qwen3.")

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
        self.model.casual_mask.add_flags(is_prefill=is_prefill)
        for layer in self.model.decoder.layers:
            if self.config.use_flash_attention:
                layer.self_attention.core_attention.add_flags(is_prefill=is_prefill)

    @jit
    def construct(self, input_ids, positions=None, batch_valid_length=None, context_lens_tensor=None, q_seq_lens=None,
                  block_tables=None, slot_mapping=None, attention_mask=None, attn_metadata=None,
                  key_cache=None, value_cache=None):
        r"""
        model forward.

        Args:
            input_ids: input ids.
            positions: position ids.
            batch_valid_length: actual seq length.
            context_lens_tensor: computed key value length.
            block_tables: Store mapping tables for each sequence.
            slot_mapping : Token cache physical slot index.
            attention_mask: attentino mask used for fa or pa.
            attn_metadata: attention metadata
            key_cache: key cache for incremental inference.
            value_cache: value cache for incremental inference.

        Returns:
            logits: the output logits.

        """
        logits = self.model(
            input_ids=input_ids,
            positions=positions,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            q_seq_lens=q_seq_lens,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            attention_mask=attention_mask,
            attn_metadata=attn_metadata,
            key_cache=key_cache,
            value_cache=value_cache
        )
        return logits

    def load_weights(self, weights_path):
        r"""
        Load weights.

        Args:
            weights_path: The path of storing weights.

        """
        rank_id = get_rank()

        source_qkv_concat = False

        sf_files = [f for f in os.listdir(weights_path) if f.endswith(".safetensors")]
        keys = []
        if sf_files:
            with safe_open(os.path.join(weights_path, sf_files[0]), framework="np") as f:
                keys = f.keys()
        for key in keys:
            if key.split('.')[-2] not in self.check_key_mapping():
                raise ValueError(f'Please enter the correct weights of safetensors')
            if key.split('.')[-2] == 'linear_qkv':
                source_qkv_concat = True
                break

        non_layer_weights, layer_weights = (self.convert_hf_weight_to_mf(weights_path))

        mf_hf_map = {}
        for weight_name in list(non_layer_weights.keys()):
            value = non_layer_weights.pop(weight_name)
            new_name = self.convert_name(weight_name)
            non_layer_weights[new_name] = value
            mf_hf_map[new_name] = weight_name
        parameter_dict = self.model.load_weights(weights_path, non_layer_weights, mf_hf_map, source_qkv_concat)
        ms.load_param_into_net(self, parameter_dict)
        del parameter_dict
        del mf_hf_map
        gc.collect()
        logger.info('................weights loading complete except the transformer layers weights................')

        num_layers = self.config.num_hidden_layers
        enable_tqdm = rank_id == 0
        with tqdm(range(num_layers), desc="Weight loading", disable=not enable_tqdm) as pbar:
            for layer_id in pbar:
                layer_weight = {}
                mf_hf_map = {}
                prefix = f"model.layers.{layer_id}."
                train_prefix = f"decoder.layers.{layer_id}."
                for weight_name in list(layer_weights.keys()):
                    if weight_name.startswith(prefix) or weight_name.startswith(train_prefix):
                        value = layer_weights.pop(weight_name)
                        new_name = self.convert_name(weight_name)
                        layer_weight[new_name] = value
                        mf_hf_map[new_name] = weight_name
                parameter_dict = self.model.load_weights(
                    weights_path, layer_weight, mf_hf_map, source_qkv_concat, layer_id)
                ms.load_param_into_net(self.model.decoder.layers[layer_id], parameter_dict)
                del parameter_dict
                gc.collect()
                pbar.set_postfix({"current_layer": layer_id})
