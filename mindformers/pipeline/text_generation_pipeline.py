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

"""TextGenerationPipeline"""
import os.path
from typing import Union, Optional

import mindspore
from mindspore import Tensor

from ..auto_class import AutoProcessor, AutoModel
from ..mindformer_book import MindFormerBook
from .base_pipeline import BasePipeline
from ..tools.register import MindFormerRegister, MindFormerModuleType
from ..models import BaseModel, BaseTokenizer, GLMForPreTraining

__all__ = ['TextGenerationPipeline']


def _setup_support_list(support_model_list):
    support_list = []
    for support_model in support_model_list:
        support_list.append(MindFormerBook.get_model_support_list().get(support_model))
    return support_list


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="text_generation")
class TextGenerationPipeline(BasePipeline):
    r"""Pipeline for Text Generation

    Args:
        model (Union[str, BaseModel]): The model used to perform task,
            the input could be a supported model name, or a model instance
            inherited from BaseModel.
        tokenizer (Optional[BaseTokenizer]): A tokenizer (None or Tokenizer)
            for text processing.

    Raises:
        TypeError: If input model and tokenizer's types are not corrected.
        ValueError: if the input model is not in support list.

    Examples:
        >>> from mindformers.pipeline import TextGenerationPipeline
        >>> text_generate = TextGenerationPipeline("gpt2")
        >>> output = text_generate("I love Beijing, because ")
    """
    _support_list = _setup_support_list(["gpt2", "glm"])
    return_name = 'text_generation'

    def __init__(self, model: Union[str, BaseModel],
                 tokenizer: Optional[BaseTokenizer] = None,
                 **kwargs):
        if isinstance(model, str):
            if model in self._support_list or os.path.isdir(model):
                if tokenizer is None:
                    tokenizer = AutoProcessor.from_pretrained(model).tokenizer
                model = AutoModel.from_pretrained(model)
                if not isinstance(tokenizer, BaseTokenizer):
                    raise TypeError(f"tokenizer should be inherited from"
                                    f" BaseTokenizer, but got {type(tokenizer)}.")
            else:
                raise ValueError(f"{model} is not supported by {self.__class__.__name__},"
                                 f"please selected from {self._support_list}.")

        if not isinstance(model, BaseModel):
            raise TypeError(f"model should be inherited from BaseModel, but got type {type(model)}.")

        # glm generate needs add_special_tokens
        if isinstance(model, GLMForPreTraining):
            kwargs['add_special_tokens'] = True

        if tokenizer is None:
            raise ValueError(f"{self.__class__.__name__}"
                             " requires for a tokenizer.")

        super().__init__(model, tokenizer, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        r"""Sanitize Parameters

        Args:
            pipeline_parameters (Optional[dict]): The parameter dict to be parsed.
        """
        preprocess_keys = ['keys', 'add_special_tokens']
        preprocess_params = {}
        for item in preprocess_keys:
            if item in pipeline_parameters:
                preprocess_params[item] = pipeline_parameters.get(item)

        postprocess_params = {}

        forward_key_name = ['top_k', 'top_p', 'do_sample', 'eos_token_id', 'repetition_penalty', 'max_length']
        forward_kwargs = {}
        for item in forward_key_name:
            if item in pipeline_parameters:
                forward_kwargs[item] = pipeline_parameters.get(item)
        return preprocess_params, forward_kwargs, postprocess_params

    def preprocess(self, inputs: Union[str, dict, Tensor],
                   **preprocess_params):
        r"""The Preprocess For Translation

        Args:
            inputs (Union[str, dict, Tensor]): The text to be classified.
            preprocess_params (dict): The parameter dict for preprocess.

        Return:
            Processed text.
        """
        add_special_tokens = preprocess_params.get('add_special_tokens', False)
        if isinstance(inputs, dict):
            keys = preprocess_params.get('keys', None)
            default_src_language_name = 'text'
            feature_name = keys.get('src_language', default_src_language_name) if keys else default_src_language_name

            inputs = inputs[feature_name]
            if isinstance(inputs, mindspore.Tensor):
                inputs = inputs.asnumpy().tolist()
        input_ids = self.tokenizer(inputs, return_tensors=None, add_special_tokens=add_special_tokens)["input_ids"]
        return {"input_ids": input_ids}

    def forward(self, model_inputs: dict,
                **forward_params):
        r"""The Forward Process of Model

        Args:
            inputs (dict): The output of preprocess.
            forward_params (dict): The parameter dict for model forward.
        """
        forward_params.pop("None", None)
        input_ids = model_inputs["input_ids"]
        output_ids = self.model.generate(input_ids, **forward_params)
        return {"output_ids": output_ids}

    def postprocess(self, model_outputs: dict,
                    **postprocess_params):
        r"""Postprocess

        Args:
            model_outputs (dict): Outputs of forward process.

        Return:
            translation results.
        """
        outputs = self.tokenizer.decode(model_outputs["output_ids"], skip_special_tokens=True)
        return [{self.return_name + '_text': outputs}]
