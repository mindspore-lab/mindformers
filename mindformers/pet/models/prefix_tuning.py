# Copyright 2024 Huawei Technologies Co., Ltd
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
"""prefix_tuning model for all llm model"""
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor

from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.pet.pet_config import PrefixTuningConfig
from mindformers.pet.tuners.prefix_tuning_adapter import PrefixTuningAdapter

class PrefixTuningModel(PreTrainedModel):
    """
    PrefixTuning Model for llm model.

    Args:
        config(PrefixTuningConfig): pet config,define parameters efficient tuning algorithm.
        base_model(PreTrainedModel): pretrained model for tuning.
    """

    def __init__(self, config: PrefixTuningConfig, base_model: PreTrainedModel):
        super().__init__(base_model.config, auto_prefix=False)
        self.config.pet_config = config
        self.pet_model = base_model
        self._check_config()
        if self.pet_model.config.is_dynamic:
            self.pet_model.set_dynamic_inputs(have_prefix_keys_values=True)
        # add prefix token
        self.prefix_layers = PrefixTuningAdapter.get_prefix(self.pet_model, self.config.pet_config)

    def _check_config(self):
        if self.pet_model.config.use_attn_mask_compression:
            raise NotImplementedError(f"PrefixTunig Method not support attn_mask_compression")
        if self.pet_model.config.fine_grain_interleave > 1:
            raise NotImplementedError(f"PrefixTunig Method not support fine_grain_interleave")

    def update_model_kwargs_before_generate(self, input_ids, model_kwargs: dict):
        return self.pet_model.update_model_kwargs_before_generate(input_ids, model_kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.pet_model.prepare_inputs_for_generation(input_ids, **kwargs)

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs = input_ids.shape[0]
        slot_mapping = Tensor(np.ones(shape=tuple([bs])), mstype.int32)
        return input_ids, labels, None, None, None, None, None, None, None, None, None, slot_mapping

    def slice_incremental_inputs(self, model_inputs: dict, current_index):
        return self.pet_model.slice_incremental_inputs(model_inputs, current_index)

    def set_dynamic_inputs(self, **kwargs):
        return self.pet_model.set_dynamic_inputs(**kwargs)

    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None,
                  zactivate_len=None, block_tables=None, slot_mapping=None):
        prefix_keys_values = self.prefix_layers(input_ids.shape[0])
        return self.pet_model(input_ids=input_ids,
                              labels=labels,
                              input_position=input_position,
                              position_ids=position_ids,
                              attention_mask=attention_mask,
                              input_embeds=input_embeds,
                              init_reset=init_reset,
                              batch_valid_length=batch_valid_length,
                              batch_index=batch_index,
                              zactivate_len=zactivate_len,
                              block_tables=block_tables,
                              slot_mapping=slot_mapping,
                              prefix_keys_values=prefix_keys_values)