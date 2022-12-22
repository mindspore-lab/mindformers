# Copyright 2022 Huawei Technologies Co., Ltd
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

"""TranslationPipeline"""
import os.path

import mindspore

from ..auto_class import AutoProcessor, AutoModel
from ..mindformer_book import MindFormerBook
from .base_pipeline import BasePipeline
from ..tools.register import MindFormerRegister, MindFormerModuleType
from ..models import BaseModel, Tokenizer

__all__ = ['TranslationPipeline']

@MindFormerRegister.register(MindFormerModuleType.PIPELINE)
class TranslationPipeline(BasePipeline):
    """
    Pipeline for translation

    Args:
        model: a pretrained model (str or BaseModel) in _supproted_list.
        tokenizer : a tokenizer (None or Tokenizer) for text processing
    """
    _support_list = MindFormerBook.get_model_support_list()['t5']
    return_name = 'translation'

    def __init__(self, model, tokenizer=None, **kwargs):
        if isinstance(model, str):
            if model in self._support_list or os.path.isdir(model):
                if tokenizer is None:
                    tokenizer = AutoProcessor.from_pretrained(model).tokenizer
                model = AutoModel.from_pretrained(model)
                if not isinstance(tokenizer, Tokenizer):
                    raise TypeError(f"tokenizer should be inherited from"
                                    f" PretrainedTokenizer, but got {type(tokenizer)}.")
            else:
                raise ValueError(f"{model} is not supported by {self.__class__.__name__},"
                                 f"please selected from {self._support_list}.")

        if not isinstance(model, BaseModel):
            raise TypeError(f"model should be inherited from BaseModel, but got type {type(BaseModel)}.")

        if tokenizer is None:
            raise ValueError(f"{self.__class__.__name__}"
                             " requires for a tokenizer.")

        super().__init__(model, tokenizer, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        """sanitize parameters for preprocess, forward, and postprocess."""
        if 'batch_size' in pipeline_parameters:
            raise ValueError(f"The {self.__class__.__name__} does not support batch inference, please remove the "
                             f"batch_size")
        preprocess_params = {}

        postprocess_params = {}

        forward_key_name = ['top_k', 'top_p', 'do_sample', 'eos_token_id', 'repetition_penalty', 'max_length']
        forward_kwargs = {}
        for item in forward_key_name:
            if item in pipeline_parameters:
                forward_kwargs[item] = pipeline_parameters.get(item)
        return preprocess_params, forward_kwargs, postprocess_params

    def preprocess(self, inputs, **preprocess_params):
        """
        Preprocess of Translation

        Args:
            inputs (url, PIL.Image, tensor, numpy): the image to be classified.
            max_length (int): max length of tokenizer's output
            padding (False / "max_length"): padding for max_length
            return_tensors ("ms"): the type of returned tensors

        Return:
            processed image.
        """
        if isinstance(inputs, dict):
            inputs = inputs['text']
            if isinstance(inputs, mindspore.Tensor):
                inputs = inputs.asnumpy().tolist()
        input_ids = self.tokenizer(inputs, return_tensors=None)["input_ids"]
        return {"input_ids": input_ids}

    def forward(self, model_inputs, **forward_params):
        """
        Forward process

        Args:
            model_inputs (dict): outputs of preprocess.

        Return:
            probs dict.
        """
        forward_params.pop("None", None)
        input_ids = model_inputs["input_ids"]
        output_ids = self.model.generate(input_ids, **forward_params)
        return {"output_ids": output_ids}

    def postprocess(self, model_outputs, **postprocess_params):
        """
        Postprocess

        Args:
            model_outputs (dict): outputs of forward process.

        Return:
            The generated results
        """
        outputs = self.tokenizer.decode(model_outputs["output_ids"], skip_special_tokens=True)
        return [{self.return_name + '_text': outputs}]
