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
"""Glm4 models' APIs."""

from typing import Dict

from mindformers.models.utils import jit
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.models.glm4.utils import Glm4PreTrainedModel
from mindformers.parallel_core.inference.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from mindformers.parallel_core.inference.model_utils import InferModelMixin
from mindformers.parallel_core.inference.quantization.utils import get_quant_config


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class InferenceGlm4ForCausalLM(Glm4PreTrainedModel, InferModelMixin):
    r"""
    Provide Glm4 model infer through network.

    Args:
        config (Glm4Config): The config of Glm4 model.

    Returns:
        output: Tensor, the output of Glm4 decoder layer

    """

    def __init__(self, config):
        super().__init__(config, auto_prefix=False)
        config: TransformerConfig = self.convert_to_transformer_config(self.config)

        self.pad_token_id = self.config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.compute_dtype = config.compute_dtype

        self.is_prefill = True
        if isinstance(self.config.parallel_decoding_params, Dict):
            self.plugin_type = self.config.parallel_decoding_params.get("plugin_type")
        else:
            self.plugin_type = None
        self.quant_config = get_quant_config(self.config, self.weight_mapping)
        self.model = GPTModel(config=config,
                              transformer_layer_spec=get_gpt_layer_local_spec(
                                  normalization=config.normalization,
                                  use_flash_attention=self.config.use_flash_attention,
                                  sandwich_norm=True,
                              ),
                              vocab_size=self.vocab_size,
                              max_sequence_length=self.max_position_embeddings,
                              rotary_percent=self.config.partial_rotary_factor,
                              rotary_base=self.config.rope_theta,
                              share_embeddings_and_output_weights=self.config.tie_word_embeddings,
                              pre_process=self.config.pre_process,
                              post_process=self.config.post_process,
                              quant_config=self.quant_config,)

    @jit
    def construct(self, input_ids, hidden_states=None, positions=None, batch_valid_length=None,
                  context_lens_tensor=None, q_seq_lens=None, block_tables=None, slot_mapping=None, attention_mask=None,
                  attn_metadata=None, attn_padding_idx=None, attn_unpadding_idx=None, ffn_padding_idx=None,
                  ffn_unpadding_idx=None, key_cache=None, value_cache=None):
        r"""
        model forward.

        Args:
            input_ids: input ids.
            positions: position ids.
            hidden_states: hidden states.
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
            hidden_states=hidden_states,
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
            value_cache=value_cache
        )
        return logits
