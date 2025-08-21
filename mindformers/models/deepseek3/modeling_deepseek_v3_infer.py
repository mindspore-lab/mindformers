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
"""Deepseek-V3 models' APIs."""
from typing import Dict

from mindspore.communication._comm_helper import _is_initialized as mindspore_comm_has_init

from mindformers.models.utils import jit
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.parallel_core.transformer_config_utils import convert_to_transformer_config
from mindformers.models.deepseek3.utils import DeepseekV3PreTrainedModel
from mindformers.parallel_core.inference.parallel_state import (
    is_initialized,
    initialize_model_parallel
)
from mindformers.parallel_core.inference.utils import update_comm_config
from mindformers.parallel_core.inference.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from mindformers.parallel_core.inference.model_utils import InferModelMixin
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs
from mindformers.models.build_config import get_quant_config
from .configuration_deepseek_v3 import DeepseekV3Config


class InferenceDeepseekV3ForCausalLM(DeepseekV3PreTrainedModel, InferModelMixin):
    r"""
    Provide Deepseek3 model infer through network.

    Args:
        config (Deepseek3Config): The config of deepseek3 model.

    Returns:
        output: Tensor, the output of qwen3 deepseek3 layer

    """

    def __init__(self, config: DeepseekV3Config):
        super().__init__(config, auto_prefix=False)
        self.config = config
        config: MLATransformerConfig = convert_to_transformer_config(
            self.config,
            is_mla_model=True,
        )
        self.transformer_config = config
        if not is_initialized() and mindspore_comm_has_init():
            initialize_model_parallel(
                data_parallel_size=config.data_parallel_size,
                tensor_model_parallel_size=config.tensor_model_parallel_size,
                expert_model_parallel_size=config.expert_model_parallel_size,
                order='tp-ep-dp',
            )
        if is_initialized():
            self.model_comm_pgs = ModelCommProcessGroups.use_parallel_state_groups(
                required_groups=['globals', 'tp', 'moe_ep', 'moe_tp', 'dp'])
        else:
            self.model_comm_pgs = default_model_comm_pgs

        # update communication-related configuration in TransformerConfig
        config = update_comm_config(config)
        self.use_fused_mla = self.config.quantization_config is not None
        config.use_fused_mla = self.use_fused_mla
        quant_config = get_quant_config(self.config.to_dict(), self.weight_mapping)
        self.pad_token_id = self.config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.compute_dtype = config.compute_dtype

        self.is_prefill = True
        if isinstance(self.config.parallel_decoding_params, Dict):
            self.plugin_type = self.config.parallel_decoding_params.get("plugin_type")
        else:
            self.plugin_type = None
        self.model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_decoder_block_spec(
                config=config,
                normalization=config.normalization,
                qk_l2_norm=False,
            ),
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_position_embeddings,
            position_embedding_type=config.position_embedding_type,
            rotary_base=config.rotary_base,
            model_comm_pgs=self.model_comm_pgs,
            quant_config=quant_config,
        )

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
            layer.self_attention.add_flags(is_prefill=is_prefill)
            layer.self_attention.core_attention.add_flags(is_prefill=is_prefill)

    @jit
    def construct(
            self,
            input_ids,
            positions=None,
            batch_valid_length=None,
            context_lens_tensor=None,
            q_seq_lens=None,
            block_tables=None,
            slot_mapping=None,
            attention_mask=None,
            attn_metadata=None,
            attn_padding_idx=None,
            attn_unpadding_idx=None,
            ffn_padding_idx=None,
            ffn_unpadding_idx=None,
            key_cache=None,
            value_cache=None
    ):
        r"""
        model forward.

        Args:
            input_ids: input ids.
            positions: position ids.
            batch_valid_length: actual seq length.
            context_lens_tensor: computed key value length.
            q_seq_lens: query sequence lengths.
            block_tables: Store mapping tables for each sequence.
            slot_mapping : Token cache physical slot index.
            attention_mask: attentino mask used for fa or pa.
            attn_metadata: attention metadata.
            attn_padding_idx: Indices mapping positions in attention output sequence to original token positions,
                used for padding attention output to fixed size.
            attn_unpadding_idx: Indices mapping valid tokens in padded attention output sequence to
                their original positions, used for removing padding in attention output.
            ffn_padding_idx: Indices mapping positions in MoE output sequence to flattened valid token positions,
                used for padding MoE output to fixed size.
            ffn_unpadding_idx: Indices mapping valid tokens in padded MoE output sequence to their original positions,
                used for removing padding in MoE output.
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
            attn_padding_idx=attn_padding_idx,
            attn_unpadding_idx=attn_unpadding_idx,
            ffn_padding_idx=ffn_padding_idx,
            ffn_unpadding_idx=ffn_unpadding_idx,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        return logits

    def convert_name(self, weight_name):
        r"""
        Override convert_name method in inference model, in order to read PTQ weights correctly.
        PTQ weights are generated after training, so it should only exist in inference model.
        """
        weight_name = super().convert_name(weight_name)
        # Do extra conversion for quantization parameters.

        # FIXME: After osl supports mcore calibration, the following conversion map should be removed.
        if self.config.quantization_config is not None:
            weight_name = weight_name.replace('model.tok_embeddings.embedding_weight',
                                              'embedding.word_embeddings.weight')
            weight_name = weight_name.replace('model.norm_out.', 'decoder.final_layernorm.')
            weight_name = weight_name.replace('lm_head.', 'output_layer.')

            weight_name = weight_name.replace('.self_attention.q_layernorm.bias',
                                              '.attention.l2q_proj.quant_op.beta')
            weight_name = weight_name.replace('.input_layernorm.bias',
                                              '.attention.q2l_proj.quant_op.beta')
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
            weight_name = weight_name.replace('.weight_offset', '.w_offset')
        if self.use_fused_mla:
            weight_name = weight_name.replace('.input_layernorm.', '.self_attention.input_layernorm.')
        return weight_name
