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
"""GPT Transformer language model"""
from typing import Literal, Optional
import gc
from tqdm import tqdm

import mindspore as ms
from mindspore import nn, ops, mint
from mindspore.communication.management import get_rank
import mindspore.common.dtype as mstype

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.parallel_core.inference.tensor_parallel.layers import ColumnParallelLinear
from mindformers.parallel_core.inference.transformer.lower_triangular_mask import LowerTriangularMaskWithDynamic
from mindformers.parallel_core.inference.base_models.common.embeddings.language_model_embedding import \
    LanguageModelEmbedding
from mindformers.parallel_core.inference.transformer.transformer_block import TransformerBlock
from mindformers.parallel_core.inference.base_models.common.embeddings.rope_utils import get_rope
from mindformers.parallel_core.inference.utils import get_tp_world_size, divide
from mindformers.tools.logger import logger


_convert_map = {
    "embedding.word_embeddings.": "split_by_tp_rank_columns",
    "self_attention.linear_qkv": "split_qkv_weight",
    "linear_qkv.bias": "split_qkv_weight",
    "self_attention.linear_proj": "split_by_tp_rank_rows",
    "mlp.linear_fc1": "split_ffn_weight",
    "mlp.linear_fc2": "split_by_tp_rank_rows",
    "input_layernorm": "not_split",
    "pre_mlp_layernorm": "not_split",
    "decoder.final_layernorm": "not_split",
    "output_layer": "split_by_tp_rank_columns",
    "self_attention.q_layernorm": "not_split",
    "self_attention.k_layernorm": "not_split",
    "router.weight.weight": "not_split",
    "experts.weight1": "split_router_expert_weight1",
    "experts.weight2": "split_router_expert_weight2",
    "self_attention.linear_qkv_down_proj": "split_linear_qkv_down_proj",
    "self_attention.linear_q_proj": "split_by_tp_rank_columns",
    "self_attention.linear_kv_down_proj": "split_linear_kv_down_proj",
    "self_attention.kv_layernorm": "not_split",
    "self_attention.linear_q_up_proj": "split_linear_q_up_proj",
    "self_attention.linear_kv_up_proj": "split_linear_kv_up_proj",
    "router.expert_bias": "not_split",
    "shared_experts.linear_fc1": "split_shared_experts",
    "shared_experts.linear_fc2": "split_by_tp_rank_rows"
}


class GPTModel(nn.Cell):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig): Configuration for the transformer model.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        vocab_size (int): Vocabulary size.
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding.
        pre_process (bool, optional): Set to true if you need to compute embedings.
            Currently only supports setting to True. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits. Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Whether to use FP16 for cross entropy,
            does not support setting to True currently. Default: False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks.
            Default: True.
        share_embeddings_and_output_weights (bool, optional): Whether to share input/output embeddings,
            does not support setting to True currently. Default: False.
        position_embedding_type (Literal['learned_absolute', 'rope', 'llama3', 'yarn', 'none'], optional):
            Type of positional embedding to use. Default: 'learned_absolute'.
        rotary_percent (float, optional): Percentage of dimensions to apply rotary embeddings. Default: 1.0.
        rotary_base (int, optional): Base value for rotary embeddings. Default: 10000.
        rope_scaling (bool, optional): Whether to use rope scaling. Default: False.
        seq_len_interpolation_factor (float, optional): Sequence length interpolation factor. Default: None.
        mtp_block_spec (ModuleSpec, optional): Specification for MTP blocks,
            does not support to set currently. Default: None.

    Inputs:
        - **input_ids** (Tensor) - Input token ids
        - **positions** (Tensor, optional) - Token positions
        - **batch_valid_length** (Tensor, optional) - Valid length of each sequence in batch
        - **context_lens_tensor** (Tensor, optional) - Context lengths tensor
        - **q_seq_lens** (Tensor, optional) - Query sequence lengths
        - **block_tables** (Tensor, optional) - Block tables for KV cache
        - **slot_mapping** (Tensor, optional) - Slot mapping for KV cache
        - **attention_mask** (Tensor, optional) - Tensor of attention mask
        - **attn_metadata** (dict, optional) - Additional attention metadata
        - **key_cache** (Tensor, optional) - Key cache for incremental inference.
        - **value_cache** (Tensor, optional) - Value cache for incremental inference.

    Outputs:
        - **output** (Tensor) - return hidden states after decoder when no post-processing

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            transformer_layer_spec: ModuleSpec,
            vocab_size: int,
            max_sequence_length: int,
            pre_process: bool = True,
            post_process: bool = True,
            fp16_lm_cross_entropy: bool = False,
            parallel_output: bool = True,
            share_embeddings_and_output_weights: bool = False,
            position_embedding_type: Literal[
                'learned_absolute', 'rope', 'llama3', 'yarn', 'none'
            ] = 'learned_absolute',
            rotary_percent: float = 1.0,
            rotary_base: int = 10000,
            rope_scaling: bool = False,
            seq_len_interpolation_factor: Optional[float] = None,
            mtp_block_spec: Optional[ModuleSpec] = None,
    ):
        super(GPTModel, self).__init__()
        if not pre_process:
            raise NotImplementedError("For GPTModel, `pre_process` is not supported to set False")
        if fp16_lm_cross_entropy:
            raise NotImplementedError("For GPTModel, `fp16_lm_cross_entropy` is not supported")
        if rope_scaling:
            raise NotImplementedError("For GPTModel, `rope_scaling` is not supported. "
                                      "Please use `rope_type` to control the selection of extrapolation algorithm.")
        if mtp_block_spec:
            raise NotImplementedError("For GPTModel, `mtp_block_spec` is not supported")

        self.config = config
        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.compute_dtype = self.config.compute_dtype

        self.max_position_embeddings = max_sequence_length
        if not hasattr(config, "qk_pos_emb_head_dim"):
            self.hidden_dim = getattr(config, "kv_channels", divide(config.hidden_size, config.num_attention_heads))
        else:
            self.hidden_dim = config.qk_pos_emb_head_dim
        self.rotary_percent = rotary_percent
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.tp_group_size = get_tp_world_size()
        self.is_prefill = True

        if hasattr(self.config, 'position_embedding_type'):
            self.position_embedding_type = self.config.position_embedding_type
        else:
            self.position_embedding_type = position_embedding_type

        if hasattr(self.config, 'rotary_base'):
            self.rotary_base = self.config.rotary_base
        else:
            self.rotary_base = rotary_base

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length
            )

        # Note: Declare the attn mask class for vLLM-MS startup
        self.casual_mask = LowerTriangularMaskWithDynamic(
            seq_length=self.config.seq_length,
            compute_type=self.config.compute_dtype,
            pad_token_id=self.config.pad_token_id,
        )

        self.rotary_pos_emb = get_rope(
            config,
            hidden_dim=self.hidden_dim,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            rotary_dtype=self.config.rotary_dtype,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            position_embedding_type=self.position_embedding_type,
            original_max_position_embeddings=self.max_position_embeddings,
        )

        # Transformer
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec
        )

        # Output
        self.output_layer = ColumnParallelLinear(
            self.config.hidden_size,
            self.vocab_size,
            config=self.config,
            bias=False,
            gather_output=self.parallel_output,
            compute_dtype=self.config.compute_dtype,
        )
        if share_embeddings_and_output_weights:
            self.output_layer.weight = self.embedding.word_embeddings.weight

        self.cast = ops.Cast()
        self.gather = ops.Gather()

    def pre_gather_func(self, output, context_lens_tensor, seq_lens_tensor):
        """Pre gather operation in infer mode."""
        if self.is_prefill:
            q_seq_lens_tensor = mint.sub(seq_lens_tensor, context_lens_tensor)
            gather_index = mint.sub(mint.cumsum(q_seq_lens_tensor, 0), 1)
            output = self.gather(output, gather_index, 0)
        return output

    def construct(self, input_ids, positions=None, batch_valid_length=None, context_lens_tensor=None,
                  q_seq_lens=None, block_tables=None, slot_mapping=None,
                  attention_mask=None, attn_metadata=None, key_cache=None, value_cache=None):
        """ Construct function of GPTModel. """

        # Generate cos and sin for RoPE.
        if self.is_prefill:
            rotary_pos_cos, rotary_pos_sin = \
                self.rotary_pos_emb.get_cos_sin_for_prefill()
        else:
            rotary_pos_cos, rotary_pos_sin = \
                self.rotary_pos_emb.get_cos_sin_for_decode(positions)

        # Decoder embedding.
        decoder_input = self.cast(self.embedding(input_ids), self.compute_dtype)

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            q_seq_lens=q_seq_lens,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            key_cache=key_cache,
            value_cache=value_cache
        )

        # Return hidden states.
        if not self.post_process:
            return hidden_states

        output = self.pre_gather_func(hidden_states, context_lens_tensor, batch_valid_length)
        # Return logits.
        logits = self.output_layer(output)
        logits = self.cast(logits.squeeze(0), mstype.float32)
        return logits

    def load_weights(self, weights_path, weight_utils):
        r"""
        The weight is processed in modules, and the weight is cut online and loaded.

        Args:
           weights_path: The path of weights.
           weight_utils: An instance of WeightsUtils.

        """
        global_rank_id = get_rank()

        weights_not_load = []

        all_weights_keys = set(weight_utils.mapping_dict.keys())

        # Weights of Transformer Layers
        enable_tqdm = global_rank_id == 0
        pbar = tqdm(range(self.config.num_layers), desc="Weight loading", disable=not enable_tqdm)
        for layer_id in pbar:
            weight_utils.parameter_dict = {}
            layer_prefixes = (f'model.layers.{layer_id}.', f'decoder.layers.{layer_id}.')
            layer_keys = []
            for k in all_weights_keys:
                for prefix in layer_prefixes:
                    if k.startswith(prefix):
                        layer_keys.append(k)
                        break
            self._deal_weight_dict(weight_utils, layer_keys, weights_path)
            _, weight_not_load = ms.load_param_into_net(self.decoder.layers[layer_id], weight_utils.parameter_dict)
            if weight_not_load is not None:
                weights_not_load.extend(weight_not_load)
            gc.collect()
            pbar.set_postfix({"current_layer": layer_id})

        # # Weights of word_embaddings, output_layer, final_layernorm
        out_layer_keys = all_weights_keys - weight_utils.processed_weights_keys
        if out_layer_keys:
            weight_utils.parameter_dict = {}
            self._deal_weight_dict(weight_utils, out_layer_keys, weights_path)
            _, weight_not_load = ms.load_param_into_net(self, weight_utils.parameter_dict)
            if weight_not_load is not None:
                weights_not_load.extend(weight_not_load)
            gc.collect()

        net_not_load = all_weights_keys - weight_utils.processed_weights_keys
        if net_not_load is not None:
            net_not_load = [weight_utils.mapping_dict[key] for key in net_not_load]
            logger.warning(f'These parameters are not loaded in the network: {net_not_load}')
        if weights_not_load is not None:
            logger.warning(f'These parameters are not loaded in the weights: {weights_not_load}')

    def _deal_weight_dict(self, weight_utils, keys, weights_path):
        """Processes and converts weight dictionary from source format to target model format.
        Args:
            weight_utils: Helper class instance containing weight conversion methods.
            keys: Source weight keys/names to be processed.
            weights_path (str): Path to the source weights file.

        """
        for weight_name in keys:
            if weight_name not in weight_utils.processed_weights_keys:
                net_name = weight_utils.mapping_dict.get(weight_name)[0]
                matched_keys = [k for k in _convert_map if k in net_name]
                matched_key = max(matched_keys, key=len) if matched_keys else None
                if matched_key:
                    src_keys = [
                        key
                        for key in weight_utils.mapping_dict
                        if weight_utils.mapping_dict[key][0] == net_name
                    ]
                    files = [weight_utils.mapping_dict[key][1] for key in src_keys]
                    src_keys_dict = dict(zip(src_keys, files))
                    getattr(weight_utils, _convert_map[matched_key])(src_keys_dict, net_name, weights_path, self.config)
                    weight_utils.processed_weights_keys.update(src_keys)
