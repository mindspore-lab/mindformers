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
"""Qwen2 models' APIs."""
__all__ = ['InferenceQwen2ForCausalLM']

from typing import Dict

from mindspore.communication._comm_helper import _is_initialized as mindspore_comm_has_init

from mindformers.models.utils import jit
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.transformer_config_utils import convert_to_transformer_config
from mindformers.models.qwen2.utils import Qwen2PreTrainedModel
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel, is_initialized
from mindformers.parallel_core.inference.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from mindformers.parallel_core.inference.model_utils import InferModelMixin
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs

from .configuration_qwen2 import Qwen2Config


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class InferenceQwen2ForCausalLM(Qwen2PreTrainedModel, InferModelMixin):
    r"""
    Provide qwen2 model infer through network.

    Args:
        config (Qwen2Config): The config of qwen2 model.

    Returns:
        output: Tensor, the output of qwen2 decoder layer

    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config, auto_prefix=False)
        self.config = config
        config: TransformerConfig = convert_to_transformer_config(self.config)
        self.transformer_config = config
        if not is_initialized() and mindspore_comm_has_init():
            initialize_model_parallel(config.tensor_model_parallel_size, order='tp')
        if is_initialized():
            self.model_comm_pgs = ModelCommProcessGroups.use_parallel_state_groups(required_groups=['tp'])
        else:
            self.model_comm_pgs = default_model_comm_pgs

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
                                  qk_layernorm=False,
                              ),
                              vocab_size=self.vocab_size,
                              max_sequence_length=self.max_position_embeddings,
                              position_embedding_type=config.position_embedding_type,
                              rotary_base=self.config.rope_theta,
                              share_embeddings_and_output_weights=self.config.tie_word_embeddings,
                              post_process=self.config.post_process,
                              model_comm_pgs=self.model_comm_pgs)

    # pylint: disable=W0613
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
            q_seq_lens: query sequence lengths.
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
