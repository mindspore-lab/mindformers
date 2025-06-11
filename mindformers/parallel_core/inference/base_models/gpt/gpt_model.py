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
from typing import Literal, Optional, Dict, Any

from mindspore import nn, ops, mint
import mindspore.common.dtype as mstype

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.parallel_core.inference.tensor_parallel.layers import ColumnParallelLinear
from mindformers.parallel_core.inference.transformer.lower_triangular_mask import LowerTriangularMaskWithDynamic
from mindformers.parallel_core.inference.transformer.rotary_embedding import (
    RotaryEmbedding,
    Llama3RotaryEmbedding,
    YaRNScalingRotaryEmbedding
)
from mindformers.parallel_core.inference.base_models.common.embeddings.language_model_embedding import \
    LanguageModelEmbedding
from mindformers.parallel_core.inference.transformer.transformer_block import TransformerBlock
from mindformers.parallel_core.inference.utils import get_tp_world_size, divide
from mindformers.parallel_core.utils.weights_utils import (deal_non_layers_weights, deal_attention_weights,
                                                           deal_moe_weights, deal_mlp_weights,
                                                           deal_mla_weights, deal_layer_norm_weights)


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
        tie_word_embeddings (bool, optional): Whether to share the input and output embedding weights. Default: False.

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
            tie_word_embeddings: Optional[bool] = False,
    ):
        super(GPTModel, self).__init__()
        if not pre_process:
            raise NotImplementedError("For GPTModel, `pre_process` is not supported to set False")
        if fp16_lm_cross_entropy:
            raise NotImplementedError("For GPTModel, `fp16_lm_cross_entropy` is not supported")
        if share_embeddings_and_output_weights:
            raise NotImplementedError("For GPTModel, `share_embeddings_and_output_weights` is not supported")
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

        self.casual_mask = LowerTriangularMaskWithDynamic(
            seq_length=self.config.seq_length,
            compute_type=self.config.compute_dtype,
            pad_token_id=self.config.pad_token_id,
        )

        self.rotary_pos_emb = self.get_rope()

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
        if tie_word_embeddings:
            self.output_layer.weight = self.embedding.word_embeddings.weight

        self.cast = ops.Cast()
        self.gather = ops.Gather()

    def pre_gather_func(self, output, context_lens_tensor, seq_lens_tensor):
        """Pre gather operation in infer mode."""
        if self.is_prefill:
            q_seq_lens_tensor = mint.sub(seq_lens_tensor, context_lens_tensor)
            gather_index = mint.sub(mint.cumsum(q_seq_lens_tensor, 0), 1)
            output = self.gather(output, gather_index, 1)
        return output

    def get_rope(self):
        """Obtain an instantiation object of RoPE class based on `position_embedding_type`"""
        # Defines the list of parameters metadata required for each RoPE type
        # When adding a new RoPE class, add the param metadata here
        extra_param_metadata = {
            'rope': [],  # no extra parameter
            'llama3': [
                'scaling_factor',
                'low_freq_factor',
                'high_freq_factor',
                'orig_max_position',
            ],
            'yarn': [
                'scaling_factor',
                'original_max_position_embeddings',
                'beta_fast',
                'beta_slow',
                'mscale',
                'mscale_all_dim',
            ],
            'none': [],  # no extra parameter
        }

        # Defines the mapping of parameters while instantiate RoPE
        mapping = {
            "scaling_factor": "rotary_scaling_factor",
            "orig_max_position": "max_position_embeddings",
            "original_max_position_embeddings": "max_position_embeddings",
        }

        # Define different RoPE class mappings
        # When adding a new RoPE class, add the mapping relationship here
        class_map: Dict[str, Any] = {
            'rope': RotaryEmbedding,
            'llama3': Llama3RotaryEmbedding,
            'yarn': YaRNScalingRotaryEmbedding,
            'none': None,
        }

        current_rope_type = self.position_embedding_type
        cls = class_map.get(current_rope_type)
        if cls is None:
            raise ValueError(f"Unsupported position embedding type: {current_rope_type}")

        required_params = extra_param_metadata.get(current_rope_type, [])
        extra_params = {
            param: getattr(self.config, mapping.get(param) if param in mapping.keys() else param, None)
            for param in required_params
        }

        missing_params = [p for p in required_params if extra_params[p] is None]
        if missing_params:
            raise ValueError(f"Missing required parameters for '{current_rope_type}': {missing_params}")

        # Base parameters for each rope
        base_params = {
            'kv_channels': self.hidden_dim,
            'rotary_percent': self.rotary_percent,
            'seq_len_interpolation_factor': self.seq_len_interpolation_factor,
            'rotary_base': self.rotary_base,
            'rotary_cos_format': 2,
            'rotary_dtype': self.config.rotary_dtype,
        }

        params = {**base_params, **extra_params}
        return cls(**params)

    def construct(self, input_ids, positions=None, batch_valid_length=None, context_lens_tensor=None,
                  q_seq_lens=None, block_tables=None, slot_mapping=None,
                  attention_mask=None, attn_metadata=None, key_cache=None, value_cache=None):
        """ Construct function of GPTModel. """
        input_ids = input_ids.reshape((1, -1))

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
            hidden_states = hidden_states.reshape((-1, hidden_states.shape[-1]))
            return hidden_states

        output = self.pre_gather_func(hidden_states, context_lens_tensor, batch_valid_length)
        # Return logits.
        logits = self.output_layer(output)
        logits = self.cast(logits.squeeze(0), mstype.float32)
        return logits

    def load_weights(self, weights_path, weights, mf_hf_map, source_qkv_concat, layer_id=None):
        r"""
        The weight is processed in modules, and the weight is cut online and loaded.

        Args:
           weights_path: The path of weights.
           weights: A dict for storing weight keys and values.
           mf_hf_map: A dict for storing keys of huggingface keys and Mindformers keys.
           source_qkv_concat: Whether to conduct qkv integration.

        Returns:
           parameter_dict: A dict that stores the divided weight.
        """
        parameter_dict = {}
        if layer_id is not None:
            layer_obj = self.decoder.layers[layer_id]
        # output_layer & word_embeddings & final_norm
        if layer_id is None:
            deal_non_layers_weights(parameter_dict, self.config, weights_path, weights, mf_hf_map)
        else:
            # attention
            deal_attention_weights(layer_obj, parameter_dict, self.config, weights_path, weights, mf_hf_map,
                                   layer_id,
                                   source_qkv_concat)

            # mlp
            deal_mlp_weights(layer_obj, parameter_dict, self.config, weights_path, weights, mf_hf_map, layer_id,
                             source_qkv_concat)

            # moe
            if self.config.num_moe_experts:
                deal_moe_weights(layer_obj, parameter_dict, self.config, weights_path, weights, mf_hf_map, layer_id)

            # mla
            deal_mla_weights(layer_obj, parameter_dict, self.config, weights_path, weights, mf_hf_map, layer_id)

            # layer_norm
            deal_layer_norm_weights(layer_obj, parameter_dict, self.config, weights_path, weights, mf_hf_map,
                                    layer_id)

        return parameter_dict
