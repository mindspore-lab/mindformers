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
from mindspore import Tensor

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.transformer_config_utils import convert_to_transformer_config
from mindformers.parallel_core.training_graph.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.training_graph.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from mindformers.parallel_core.utils.model_mixin import TrainModelMixin
from mindformers.models.qwen3.utils import Qwen3PreTrainedModel
from .configuration_qwen3 import Qwen3Config


class TrainingQwen3ForCausalLM(Qwen3PreTrainedModel, TrainModelMixin):
    r"""
    Provide qwen2 model infer through network.

    Args:
        config (Qwen3Config): The config of qwen3 model.

    Returns:
        output: Tensor, the output of qwen3 decoderlayer

    """
    def __init__(self, config: Qwen3Config):
        super().__init__(config, auto_prefix=False)
        config: TransformerConfig = convert_to_transformer_config(self.config)

        self.model = GPTModel(config=config,
                              transformer_layer_spec=get_gpt_layer_local_spec(
                                  qk_layernorm=True,
                                  use_contiguous_weight_layout=config.use_contiguous_weight_layout
                              ),
                              vocab_size=config.vocab_size,
                              max_sequence_length=config.max_position_embeddings,
                              position_embedding_type=config.position_embedding_type,
                              rotary_base=self.config.rope_theta,
                              share_embeddings_and_output_weights=self.config.tie_word_embeddings,
                              post_process=self.config.post_process)

    def construct(
            self,
            input_ids: Tensor,
            position_ids: Tensor = None,
            attention_mask: Tensor = None,
            decoder_input: Tensor = None,
            labels: Tensor = None,
            extra_block_kwargs=None,
            prefix_keys_values=None,
            loss_mask=None,
            actual_seq_len=None
    ):
        """Qwen3 construct for training"""
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            extra_block_kwargs=extra_block_kwargs,
            prefix_keys_values=prefix_keys_values,
            loss_mask=loss_mask,
            actual_seq_len=actual_seq_len
        )
