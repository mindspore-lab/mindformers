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
import re
from tqdm import tqdm

import mindspore as ms
from mindspore import nn, ops, mint
import mindspore.common.dtype as mstype

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.parallel_core.inference.tensor_parallel.layers import ColumnParallelLinear
from mindformers.parallel_core.inference.transformer.lower_triangular_mask import LowerTriangularMaskWithDynamic
from mindformers.parallel_core.inference.base_models.common.embeddings.language_model_embedding import \
    LanguageModelEmbedding
from mindformers.parallel_core.inference.transformer.transformer_block import TransformerBlock
from mindformers.parallel_core.inference.base_models.common.embeddings.rope_utils import get_rope
from mindformers.parallel_core.inference.utils import divide
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs
from mindformers.tools.logger import logger


_convert_map = {
    ("embedding.word_embeddings", "output_layer", "self_attention.linear_q_proj"): "split_by_tp_rank_columns",

    ("self_attention.linear_qkv", "linear_qkv.bias",
     "mlp.linear_fc1", "shared_experts.linear_fc1"): "add_qkv_ffn_weight_into_dict",

    ("self_attention.linear_proj", "mlp.linear_fc2", "shared_experts.linear_fc2"): "split_by_tp_rank_rows",

    ("input_layernorm", "pre_mlp_layernorm", "decoder.final_layernorm",
     "self_attention.q_layernorm", "self_attention.k_layernorm",
     "router.weight.weight", "self_attention.kv_layernorm",
     "router.expert_bias"): "not_split",

    ("experts.weight1",): "add_router_expert_weight1_into_dict",

    ("experts.weight2",): "add_router_expert_weight2_into_dict",

    ("self_attention.linear_qkv_down_proj", "self_attention.linear_kv_down_proj"): "add_linear_kv_down_proj_into_dict",

    ("self_attention.linear_q_up_proj",): "add_linear_q_up_proj_into_dict",

    ("self_attention.linear_kv_up_proj",): "add_linear_kv_up_proj_into_dict",
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
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.

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
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
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
        self.tp = model_comm_pgs.tp
        self.tp_group_size = self.tp.size
        self.tp_rank = self.tp.rank
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
                max_sequence_length=self.max_sequence_length,
                model_comm_pgs=model_comm_pgs,
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
            spec=transformer_layer_spec,
            model_comm_pgs=model_comm_pgs,
        )

        # Output
        self.output_layer = ColumnParallelLinear(
            self.config.hidden_size,
            self.vocab_size,
            config=self.config,
            bias=False,
            gather_output=self.parallel_output,
            compute_dtype=self.config.compute_dtype,
            tp_group=model_comm_pgs.tp,
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
                self.rotary_pos_emb.get_cos_sin_for_prefill(self.max_position_embeddings)
        else:
            rotary_pos_cos, rotary_pos_sin = \
                self.rotary_pos_emb.get_cos_sin_for_decode(positions, self.max_position_embeddings)

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

    def load_weights(self, weights_loader):
        r"""
        The weight is processed in modules, and the weight is cut online and loaded.

        Args:
           weights_loader: An instance of WeightsUtils.

        """
        network_not_load = []
        weights_not_load = []

        all_weights_keys = set(weights_loader.mapping_dict.keys())
        warned_layers = set()
        pattern = re.compile(r'\.')

        with tqdm(total=len(all_weights_keys), desc="Loading weights") as pbar:
            while all_weights_keys:
                weight_key = next(iter(all_weights_keys))
                if not (weight_key.endswith("weight") or weight_key.endswith("bias")):
                    all_weights_keys.remove(weight_key)
                    weights_not_load.append(weight_key)
                    pbar.update(1)
                    continue
                parts = pattern.split(weight_key)
                layer_id = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
                if layer_id is not None:
                    if layer_id >= self.config.num_layers:
                        if layer_id not in warned_layers:
                            logger.warning(f'Layer {layer_id} exceeds network depth, skipping weights.')
                            warned_layers.add(layer_id)
                        all_weights_keys.remove(weight_key)
                        pbar.update(1)
                        continue
                    layer_prefixes = (f'model.layers.{layer_id}.', f'decoder.layers.{layer_id}.')
                    layer_keys = {k for k in all_weights_keys if k.startswith(layer_prefixes)}
                    self._deal_weight_dict(weights_loader, layer_keys, weights_not_load)
                    net_not_load, _ = ms.load_param_into_net(
                        self.decoder.layers[layer_id], weights_loader.parameter_dict)
                    if net_not_load:
                        network_not_load.append(net_not_load)
                    all_weights_keys -= layer_keys
                    gc.collect()
                    pbar.update(len(layer_keys))
                    pbar.set_postfix({"current": f"layer_{layer_id}"})

                else:
                    net_name = weights_loader.mapping_dict.get(weight_key)[0]
                    matched_key = self.find_matching_key(net_name)
                    if matched_key == 'decoder.final_layernorm':
                        self._update_weight_dict(matched_key, weight_key, weights_loader)
                        net_not_load, _ = ms.load_param_into_net(
                            self.decoder.final_layernorm, weights_loader.parameter_dict)
                        if net_not_load:
                            network_not_load.append(net_not_load)
                    elif matched_key == 'output_layer':
                        self._update_weight_dict(matched_key, weight_key, weights_loader)
                        net_not_load, _ = ms.load_param_into_net(
                            self.output_layer, weights_loader.parameter_dict)
                        if net_not_load:
                            network_not_load.append(net_not_load)
                    elif matched_key == 'embedding.word_embeddings':
                        self._update_weight_dict(matched_key, weight_key, weights_loader)
                        net_not_load, _ = ms.load_param_into_net(
                            self.embedding.word_embeddings, weights_loader.parameter_dict)
                        if net_not_load:
                            network_not_load.append(net_not_load)
                    else:
                        weights_not_load.append(weight_key)

                    all_weights_keys.remove(weight_key)
                    gc.collect()
                    pbar.update(1)
                    pbar.set_postfix({"current": matched_key or "other"})

            logger.warning(f'These parameters are not loaded in the network: {network_not_load}')
            logger.warning(f'These parameters are not loaded in the weights: {weights_not_load}')

    def _update_weight_dict(self, matched_key, weight_key, weights_loader):
        net_name = weights_loader.mapping_dict.get(weight_key)[0]
        file = weights_loader.mapping_dict.get(weight_key)[1]
        src_keys_dict = {weight_key: file}
        func = next((v for k, v in _convert_map.items() if matched_key in k), None)
        getattr(weights_loader, func)(src_keys_dict, net_name, self.config)

    def _deal_weight_dict(self, weights_loader, keys, weights_not_load):
        """Process weight dictionary and load matching weights into the model.

        Args:
            weights_loader (object): Helper object that contains weight loading utilities and mapping information
            keys (list): List of weight names to be processed
            weights_not_load (list): Output list that will contain names of weights that failed to load

        """
        processed_weights_keys = set()
        for weight_name in keys:
            if weight_name not in processed_weights_keys:
                net_name = weights_loader.mapping_dict.get(weight_name)[0]
                matched_key = self.find_matching_key(net_name)
                if matched_key:
                    src_keys = [
                        key
                        for key in weights_loader.mapping_dict
                        if weights_loader.mapping_dict[key][0] == net_name
                    ]
                    files = [weights_loader.mapping_dict[key][1] for key in src_keys]
                    src_keys_dict = dict(zip(src_keys, files))
                    func = next((v for k, v in _convert_map.items() if matched_key in k), None)
                    getattr(weights_loader, func)(src_keys_dict, net_name, self.config)
                    processed_weights_keys.update(src_keys)
                else:
                    weights_not_load.append(weight_name)

    def find_matching_key(self, net_name):
        matched_keys = []
        for key_tuple in _convert_map:
            for pattern in key_tuple:
                if pattern in net_name:
                    matched_keys.append(pattern)
        if matched_keys:
            return max(matched_keys, key=len)
        return None
