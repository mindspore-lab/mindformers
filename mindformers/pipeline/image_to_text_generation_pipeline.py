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
"""Image Classification Pipeline API."""
from typing import Optional, Union

import numpy as np
from PIL import Image
from mindspore import Tensor, Model
from mindspore.ops import operations as P

from mindformers.mindformer_book import MindFormerBook
from mindformers.models import PreTrainedModel, BaseImageProcessor
from mindformers.tools.image_tools import load_image
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from .base_pipeline import Pipeline

__all__ = ['ImageToTextPipeline']


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="image_to_text_generation")
class ImageToTextPipeline(Pipeline):
    r"""Pipeline for image to text generation

    Args:
        model (Union[PretrainedModel, Model]):
            The model used to perform task, the input should be a model instance inherited from PretrainedModel.
        image_processor (Optional[BaseImageProcessor]):
            The image_processor of model, it could be None if the model do not need image_processor.

    Raises:
        TypeError:
            If input model and image_processor's types are not corrected.
        ValueError:
            If the input model is not in support list.
    """
    _support_list = MindFormerBook.get_pipeline_support_task_list()['image_to_text_generation'].keys()

    def __init__(self, model: Union[PreTrainedModel, Model],
                 image_processor: Optional[BaseImageProcessor] = None,
                 tokenizer=None,
                 **kwargs):

        if image_processor is None:
            raise ValueError("ImageToTextPipeline"
                             " requires for a image_processor.")
        self.hypothesis_template = kwargs.pop("hypothesis_template", "{}")
        super().__init__(model, image_processor=image_processor, tokenizer=tokenizer, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        r"""Sanitize Parameters

        Args:
            pipeline_parameters (Optional[dict]):
                The parameter dict to be parsed.
        """
        preprocess_params = {}
        postprocess_params = {}
        forward_params = {}

        post_list = ["top_k"]
        pre_list = ["hypothesis_template", "max_length", "padding"]
        forward_list = ['top_k', 'top_p', 'do_sample', 'eos_token_id', 'repetition_penalty', 'max_length', 'seed']
        for item in post_list:
            if item in pipeline_parameters:
                postprocess_params[item] = pipeline_parameters.get(item)

        for item in pre_list:
            if item in pipeline_parameters:
                preprocess_params[item] = pipeline_parameters.get(item)

        for item in forward_list:
            if item in pipeline_parameters:
                forward_params[item] = pipeline_parameters.get(item)

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, inputs: (Union[str, dict, Image.Image, Tensor, np.ndarray]),
                   **preprocess_params):
        r"""The Preprocess For Task

        Args:
            inputs (Union[url, dict, PIL.Image, tensor, numpy]):
                Inputs used to generate text, including image, and prompt (if provided).
            preprocess_params (dict):
                The parameter dict for preprocess.

        Return:
            Processed image and prompt.
        """
        if isinstance(inputs, dict):
            image = inputs['image']
            prompt = inputs.get('prompt', None)
        else:
            image = inputs
            prompt = ""

        if isinstance(image, str):
            image = load_image(image)
        image_processed = self.image_processor(image)

        max_length = preprocess_params.pop("max_length", 32)
        padding = preprocess_params.pop("padding", "max_length")
        hypothesis_template = preprocess_params.pop("hypothesis_template", None)
        if hypothesis_template is not None:
            prompt = hypothesis_template.format(prompt).strip()
        else:
            prompt = self.hypothesis_template.format(prompt).strip()

        if not prompt:
            prompt = self.tokenizer.pad_token

        prompt_processed = self.tokenizer(prompt,
                                          max_length=max_length,
                                          padding=padding,
                                          return_tensors="ms",
                                          add_special_tokens=False,
                                          **preprocess_params)
        prompt_input_ids = prompt_processed["input_ids"]

        if len(prompt_input_ids.shape) == 1:
            prompt_input_ids = P.ExpandDims()(prompt_input_ids, 0)

        return {"image_processed": image_processed, "prompt_input_ids": prompt_input_ids}

    def _forward(self, model_inputs: dict,
                 **forward_params):
        r"""The Forward Process of Model

        Args:
            model_inputs (dict):
                The output of preprocess.
            forward_params (dict):
                The parameter dict for model forward.
        """
        image_processed = model_inputs["image_processed"]
        prompt_input_ids = model_inputs["prompt_input_ids"]

        output_ids_per_image = self.network.generate_text_for_image(image_processed, prompt_input_ids)
        return {"output_ids": output_ids_per_image}

    def postprocess(self, model_outputs, **postprocess_params):
        output_ids = model_outputs["output_ids"]

        outputs = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return outputs
