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

from mindformers.models.modeling_utils import PreTrainedModel

from mindformers.pet.pet_config import LoraConfig
from mindformers.pet.tuners.lora_adapter import LoraAdapter
from mindformers.tools import logger


class LoraModel(PreTrainedModel):
    """
    Lora Model for llm model.

    Args:
        config(LoraConfig): pet config,define parameters efficient tuning algorithm.
        base_model(PreTrainedModel): pretrained model for tuning.
    """
    def __init__(self, config: LoraConfig, base_model: PreTrainedModel):
        super().__init__(base_model.config, auto_prefix=False)
        self.config.pet_config = config
        self._check_config()
        # add lora layer.
        self.lora_model = self.add_adapter(base_model)

    def add_adapter(self, base_model: PreTrainedModel):
        """Add adapter for layers."""
        if hasattr(base_model, "backbone"):
            base_model.backbone = LoraAdapter.get_pet_model(base_model.backbone, self.config.pet_config)
        elif hasattr(base_model, "model"):
            base_model.model = LoraAdapter.get_pet_model(base_model.model, self.config.pet_config)
        elif hasattr(base_model, "transformer"):
            base_model.transformer = LoraAdapter.get_pet_model(base_model.transformer, self.config.pet_config)
        else:
            logger.warning("The base model must has an attribute named in \'backbone\',"
                           "\'model\', or \'transformer\', which define transformer blocks.")
        return base_model

    def _check_config(self):
        if self.config.pet_config.target_modules is None:
            raise ValueError(f"No target modules for lora layer.")

    def update_model_kwargs_before_generate(self, input_ids, model_kwargs: dict):
        return self.lora_model.update_model_kwargs_before_generate(input_ids, model_kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.lora_model.prepare_inputs_for_generation(input_ids, **kwargs)

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        return self.lora_model.prepare_inputs_for_predict_layout(input_ids, **kwargs)

    def slice_incremental_inputs(self, model_inputs: dict, current_index):
        return self.lora_model.slice_incremental_inputs(model_inputs, current_index)

    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None):
        return  self.lora_model(input_ids=input_ids,
                                labels=labels,
                                input_position=input_position,
                                position_ids=position_ids,
                                attention_mask=attention_mask,
                                input_embeds=input_embeds,
                                init_reset=init_reset,
                                batch_valid_length=batch_valid_length)
