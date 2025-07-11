# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Lora model for all llm model"""
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor

from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.pet.pet_config import LoraConfig
from mindformers.pet.tuners.lora_adapter import LoraAdapter
from mindformers.tools.logger import logger


class LoraModel(PreTrainedModel):
    r"""
    LoRA model for LLM. Provide a flexible and efficient way to adjust and
    optimize pre-trained models by adding LoRA structures to the base pre-trained models.

    Args:
        config (LoraConfig): Pet config, defines Parameter-Efficient Tuning (Pet) algorithm.
        base_model (PreTrainedModel): Pre-trained model for tuning.

    Inputs:
        - **\*inputs** (Tensor) - Input arguments of original base model.

    Outputs:
        Outputs of original base model.

    Examples:
        >>> import mindspore as ms
        >>> from mindformers.pet import LoraModel, LoraConfig
        >>> from mindformers.models import LlamaConfig, LlamaForCausalLM
        >>> ms.set_context(mode=0)
        >>> config = LlamaConfig(num_layers=2)
        >>> lora_config = LoraConfig(target_modules='.*wq|.*wk|.*wv|.*wo')
        >>> model = LlamaForCausalLM(config)
        >>> lora_model = LoraModel(lora_config,model)
        >>> print(lora_model.lora_model)
        LlamaForCausalLM<
        (model): LlamaModel<
        (freqs_mgr): FreqsMgr<>
        (casual_mask): LowerTriangularMaskWithDynamic<>
        (tok_embeddings): LlamaEmbedding<>
        (layers): CellList<
        (0): LLamaDecodeLayer<
        (ffn_norm): LlamaRMSNorm<>
        (attention_norm): LlamaRMSNorm<>
        (attention): LLamaAttention<
        (wq): LoRADense<
        input_channels=4096, output_channels=4096
        (lora_dropout): Dropout<p=0.01>
        >
        (wk): LoRADense<
        input_channels=4096, output_channels=4096
        (lora_dropout): Dropout<p=0.01>
        >
        (wv): LoRADense<
        input_channels=4096, output_channels=4096
        (lora_dropout): Dropout<p=0.01>
        >
        (wo): LoRADense<
        input_channels=4096, output_channels=4096
        (lora_dropout): Dropout<p=0.01>
        >
        (apply_rotary_emb): RotaryEmbedding<>
        >
        (feed_forward): LlamaFeedForward<
        (w1): Linear<
        (activation): LlamaSiLU<>
        >
        (w2): Linear<>
        (w3): Linear<>
        >
        >
        (1): LLamaDecodeLayer<
        (ffn_norm): LlamaRMSNorm<>
        (attention_norm): LlamaRMSNorm<>
        (attention): LLamaAttention<
        (wq): LoRADense<
        input_channels=4096, output_channels=4096
        (lora_dropout): Dropout<p=0.01>
        >
        (wk): LoRADense<
        input_channels=4096, output_channels=4096
        (lora_dropout): Dropout<p=0.01>
        >
        (wv): LoRADense<
        input_channels=4096, output_channels=4096
        (lora_dropout): Dropout<p=0.01>
        >
        (wo): LoRADense<
        input_channels=4096, output_channels=4096
        (lora_dropout): Dropout<p=0.01>
        >
        (apply_rotary_emb): RotaryEmbedding<>
        >
        (feed_forward): LlamaFeedForward<
        (w1): Linear<
        (activation): LlamaSiLU<>
        >
        (w2): Linear<>
        (w3): Linear<>
        >
        >
        >
        (norm_out): LlamaRMSNorm<>
        >
        (lm_head): Linear<>
        (loss): CrossEntropyLoss<
        (_log_softmax): _LogSoftmax<>
        (_nllloss): _NLLLoss<>
        >
        >
    """

    def __init__(self, config: LoraConfig, base_model: PreTrainedModel):
        super().__init__(base_model.config, auto_prefix=False)
        self.config.pet_config = config
        self._check_config()
        # add lora layer.
        self.network = self.add_adapter(base_model)
        self.network.update_parameters_name()

    def add_adapter(self, base_model: PreTrainedModel):
        """Add adapter for layers."""
        if hasattr(base_model, "network"):
            base_model.network = LoraAdapter.get_pet_model(base_model.network, self.config.pet_config)
        elif hasattr(base_model, "backbone"):
            base_model.backbone = LoraAdapter.get_pet_model(base_model.backbone, self.config.pet_config)
        elif hasattr(base_model, "model"):
            base_model.model = LoraAdapter.get_pet_model(base_model.model, self.config.pet_config)
        elif hasattr(base_model, "transformer"):
            base_model.transformer = LoraAdapter.get_pet_model(base_model.transformer, self.config.pet_config)
        elif hasattr(base_model, "llm_model"):
            base_model.llm_model = LoraAdapter.get_pet_model(base_model.llm_model, self.config.pet_config)
        elif hasattr(base_model, "language_model"):
            base_model.language_model = LoraAdapter.get_pet_model(base_model.language_model, self.config.pet_config)
        else:
            logger.warning("The base model must has an attribute named in \'backbone\',"
                           "\'model\', or \'transformer\', which define transformer blocks.")
        return base_model

    def _check_config(self):
        if self.config.pet_config.target_modules is None:
            raise ValueError(f"No target modules for lora layer.")

    def update_model_kwargs_before_generate(self, input_ids, model_kwargs: dict):
        return self.network.update_model_kwargs_before_generate(input_ids, model_kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.network.prepare_inputs_for_generation(input_ids, **kwargs)

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs, seq = input_ids.shape[0], input_ids.shape[1]
        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        return input_ids, labels, None, None, None, None, None, None, None, None, None, slot_mapping

    def slice_incremental_inputs(self, model_inputs: dict, current_index):
        return self.network.slice_incremental_inputs(model_inputs, current_index)

    def set_dynamic_inputs(self, **kwargs):
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                        dynamic_batch_valid_length, None, None, dynamic_block_tables, dynamic_slot_mapping)

    def to_embeddings(self, tokens):
        return self.network.to_embeddings(tokens)

    def convert_name(self, weight_name):
        return self.network.convert_name(weight_name)

    def convert_weight_dict(self, source_dict, **kwargs):
        return self.network.convert_weight_dict(source_dict, **kwargs)

    def convert_map_dict(self, source_dict, **kwargs):
        return self.network.convert_map_dict(source_dict, **kwargs)

    def obtain_name_map(self, load_checkpoint_files):
        return self.network.obtain_name_map(load_checkpoint_files)

    def construct(self, *inputs, **kwargs):
        return self.network(*inputs, **kwargs)
