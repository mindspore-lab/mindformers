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
"""Multi-Modal to Text Generation Pipeline API."""
from typing import Optional, Union, Dict, List

from mindspore import Model
from mindformers.mindformer_book import MindFormerBook
from mindformers.models import PreTrainedModel, BaseXModalToTextProcessor
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from .base_pipeline import Pipeline

__all__ = ['MultiModalToTextPipeline']


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="multi_modal_to_text_generation")
class MultiModalToTextPipeline(Pipeline):
    r"""Pipeline for multi-modal to text generation

    Args:
        model (Union[PretrainedModel, Model]):
            The model used to perform task, the input should be a model instance inherited from PretrainedModel.
        processor (Optional[BaseXModalToTextProcessor]):
            The image_processor of model, it could be None if the model do not need image_processor.

    Raises:
        TypeError:
            If input model and image_processor's types are not corrected.
        ValueError:
            If the input model is not in support list.
    """
    _support_list = MindFormerBook.get_pipeline_support_task_list()['multi_modal_to_text_generation'].keys()

    def __init__(self, model: Union[PreTrainedModel, Model],
                 processor: Optional[BaseXModalToTextProcessor] = None,
                 **kwargs):
        if processor is None:
            raise ValueError("MultiModalToTextPipeline requires a processor.")
        self.processor = processor
        self.output_columns = self.processor.output_columns
        super().__init__(model, processor=processor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        r"""Sanitize Parameters

        Args:
            pipeline_parameters (Optional[dict]):
                The parameter dict to be parsed.
        """
        preprocess_params = {}
        postprocess_params = {}
        forward_params = {}
        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, inputs: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
                   **preprocess_params):
        r"""The Preprocess For Task

        Args:
            inputs (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]):
                Inputs used to generate text.
            preprocess_params (dict):
                The parameter dict for preprocess.

        Return:
            Processed image, input_ids, img_pos.
        """
        process_res = self.processor(query_list=inputs)
        if "raw_query" in process_res:
            process_res.pop("raw_query")

        return process_res

    def _forward(self, model_inputs: dict,
                 **forward_params):
        r"""The Forward Process of Model

        Args:
            model_inputs (dict):
                The output of preprocess.
            forward_params (dict):
                The parameter dict for model forward.
        """
        output_ids = self.network.generate(**model_inputs, **forward_params)
        return {"output_ids": output_ids}

    def postprocess(self, model_outputs, **postprocess_params):
        output_ids = model_outputs["output_ids"]
        return self.processor.post_process(output_ids, **postprocess_params)
