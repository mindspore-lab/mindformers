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

"""
BaseXModalToTextModel
"""
import abc

import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.ops import operations as P

from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.tools.logger import logger
from mindformers.utils import deprecated


@deprecated
class BaseXModalToTextModel(PreTrainedModel):
    """Base modal of multi-modal to text """
    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()

        self.tensor_scatter_update = ops.TensorScatterUpdate().shard(((1, 1, 1),
                                                                      (1, 1, 1),
                                                                      (1, 1, 1)))

    def check_pipeline_stage(self):
        """check pipeline_stage and num_layers"""
        config = self.config
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        pp = config.parallel_config.pipeline_stage
        if parallel_mode in ["semi_auto_parallel"] and pp > 1:
            logger.warning("No overloading check_pipeline_stage function is found while training multi-modal model. "
                           "Use num_layers = config.vision_model.num_hidden_layers + config.llm_model.num_layers.")
            num_layers = config.vision_model.model_config.num_hidden_layers + config.llm_model.model_config.num_layers
            if num_layers < pp:
                raise ValueError(
                    f"num_layers of model should be greater than or equal to pipeline_stage, "
                    f"but get num_layers ({num_layers}) < pp({pp})"
                )
            pipeline_interleave_enabled = ms.get_auto_parallel_context("pipeline_interleave")
            pp_interleave_num = getattr(config, 'pp_interleave_num', 0) or 0
            if pipeline_interleave_enabled and pp_interleave_num * pp > num_layers:
                raise ValueError(
                    f"num_layers should be greater than `pp * pp_interleave_num`, "
                    f"but got num_layers : {num_layers} "
                    f"and pp * pp_interleave_num = {pp * pp_interleave_num}."
                )

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        raise NotImplementedError("this method needs to be implemented in multi-card inference")

    def update_modal_to_text(self, modal_embeds: Tensor, text_embeds: Tensor, modal_context_pos: Tensor):
        """update the value at a specific position of the text embedding with the modal embeddings"""
        modal_embeds = self.cast(modal_embeds, text_embeds.dtype)
        text_embeds = self.tensor_scatter_update(text_embeds, modal_context_pos, modal_embeds)
        return text_embeds

    @abc.abstractmethod
    def kvcache(self, layer_idx):
        raise NotImplementedError("this method needs to be implemented in kbk inference")

    @abc.abstractmethod
    def set_dynamic_inputs(self, **kwargs):
        raise NotImplementedError("this method needs to be implemented in kbk inference and is_dynamic=True")
