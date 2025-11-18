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
"""Telechat2 models' APIs."""
__all__ = ['InferenceTelechat2ForCausalLM']

from typing import Any, Generator, Tuple, List
from safetensors import safe_open
from tqdm.auto import tqdm
import numpy as np

from mindformers.models.utils import jit
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.models.telechat2.utils import Telechat2PreTrainedModel
from mindformers.parallel_core.inference.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from mindformers.parallel_core.inference.model_utils import InferModelMixin
from mindformers.parallel_core.inference.quantization.utils import get_quant_config
from mindformers.parallel_core.inference.parallel_state import get_tensor_model_parallel_rank

from .configuration_telechat2 import Telechat2Config


class InferenceTelechat2ForCausalLM(Telechat2PreTrainedModel, InferModelMixin):
    r"""
    Provide telechat2 model infer through network.

    Args:
        config (Telechat2Config): The config of telechat2 model.

    Returns:
        output: Tensor, the output of telechat2 decoder layer

    """

    def __init__(self, config: Telechat2Config):
        super().__init__(config, auto_prefix=False)
        self.config = config
        config: TransformerConfig = self.convert_to_transformer_config(self.config)

        self.pad_token_id = self.config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.compute_dtype = config.compute_dtype
        self.quant_config = get_quant_config(self.config, self.weight_mapping)

        self.is_prefill = True
        self.model = GPTModel(config=config,
                              transformer_layer_spec=get_gpt_layer_local_spec(
                                  normalization=config.normalization,
                                  use_flash_attention=self.config.use_flash_attention,
                                  qk_layernorm=False,
                              ),
                              vocab_size=self.vocab_size,
                              max_sequence_length=self.max_position_embeddings,
                              position_embedding_type=config.position_embedding_type,
                              share_embeddings_and_output_weights=self.config.tie_word_embeddings,
                              pre_process=config.pre_process,
                              post_process=config.post_process,
                              quant_config=self.quant_config)

    @jit
    def construct(self, input_ids, hidden_states=None, positions=None, batch_valid_length=None,
                  context_lens_tensor=None, q_seq_lens=None, block_tables=None, slot_mapping=None,
                  attention_mask=None, attn_metadata=None, attn_padding_idx=None, attn_unpadding_idx=None,
                  ffn_padding_idx=None, ffn_unpadding_idx=None, key_cache=None, value_cache=None):
        """
        model forward.

        Args:
            input_ids: input ids.
            positions: position ids.
            hidden_states: hidden states.
            batch_valid_length: actual seq length.
            context_lens_tensor: computed key value length.
            q_seq_lens: query sequence lengths.
            block_tables: Store mapping tables for each sequence.
            slot_mapping : Token cache physical slot index.
            attention_mask: attention mask used for fa or pa.
            attn_metadata: attention metadata
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
            hidden_states=hidden_states,
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

    def convert_name(self, weight_name):
        r"""
        Override convert_name method in inference model, in order to read PTQ weights correctly.
        PTQ weights are generated after training, so it should only exist in inference model.
        """
        weight_name = super().convert_name(weight_name)
        # Do extra conversion for quantization parameters.

        # After osl supports mcore calibration, the following conversion map should be removed.
        if self.config.quantization is not None:
            weight_name = weight_name.replace('model.decoder.layers.', 'decoder.layers.')
            weight_name = weight_name.replace('model.word_embeddings.', 'embedding.word_embeddings.')
            weight_name = weight_name.replace('model.embedding.word_embeddings.', 'embedding.word_embeddings.')
            weight_name = weight_name.replace('model.output_layer.', 'output_layer.')
            weight_name = weight_name.replace('model.decoder.final_layernorm.', 'decoder.final_layernorm.')
        return weight_name

    def _safetensors_weights_iterator(self, weights_files: List[str]) -> Generator[Tuple[str, Any], None, None]:
        """Iterate over the weights in the model safetensor files."""
        rank_id = get_tensor_model_parallel_rank()
        is_main_rank = rank_id == 0
        for st_file in tqdm(
                weights_files,
                desc=f"[Rank {rank_id}] Loading safetensors checkpoint shards",
                disable=not is_main_rank
        ):
            with safe_open(st_file, framework="np") as f:
                for name in f.keys():  # noqa: SIM118
                    # Return a lightweight PySafeSlice object
                    # that uses file pointer offset internally to read Safetensor
                    # on demand, avoiding memory explosion. Actual data can be obtained through slicing operation
                    # like param[start:end]
                    param = f.get_slice(name)
                    name = self.convert_name(name)
                    if ".key_value." in name and self.quant_config is None and self.config.quantization is None:
                        num_heads = self.config.n_head
                        n_kv_heads = self.config.num_key_value_heads
                        hidden_size = self.config.hidden_size
                        head_dim = hidden_size // num_heads
                        n_rep = num_heads // n_kv_heads
                        kv_channel = hidden_size // n_rep
                        param = np.array(param[:])
                        param = param.reshape((n_kv_heads, 2*head_dim, -1))
                        k_key = name.replace("key_value", "linear_k")
                        k_weight = param[:, :head_dim, :].reshape((kv_channel, -1))
                        v_key = name.replace("key_value", "linear_v")
                        v_weight = param[:, head_dim:2*head_dim, :].reshape((kv_channel, -1))
                        yield k_key, k_weight
                        yield v_key, v_weight
                    else:
                        yield name, param
