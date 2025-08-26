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
"""Qwen3Moe models' APIs."""

__all__ = ["TrainingQwen3MoeForCausalLM"]
from mindspore import Tensor

from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.models.qwen3_moe.utils import Qwen3MoePreTrainedModel


from mindformers.parallel_core.training_graph.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.training_graph.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from mindformers.parallel_core.utils.model_mixin import TrainModelMixin


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class TrainingQwen3MoeForCausalLM(Qwen3MoePreTrainedModel, TrainModelMixin):
    r"""
    Provide qwen3_moe model infer through network.

    Args:
        config (Qwen3MoeConfig): The config of qwen3_moe model.

    Returns:
        output: Tensor, the output of qwen3_moe decoder layer

    """

    def __init__(self, config):
        super().__init__(config, auto_prefix=False)
        config: TransformerConfig = self.convert_to_transformer_config(self.config)

        self.model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_local_spec(
                moe_grouped_gemm=config.moe_grouped_gemm,
                num_experts=config.num_moe_experts,
                use_contiguous_weight_layout_attention=config.use_contiguous_weight_layout_attention,
                qk_layernorm=True,
                use_interleaved_weight_layout_mlp=config.use_interleaved_weight_layout_mlp,
            ),
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_position_embeddings,
            position_embedding_type=config.position_embedding_type,
            rotary_base=self.config.rope_theta,
            share_embeddings_and_output_weights=self.config.tie_word_embeddings,
            post_process=self.config.post_process,
        )

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
            actual_seq_len=None,
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
            actual_seq_len=actual_seq_len,
        )
