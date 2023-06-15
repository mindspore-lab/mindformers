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
# from tkinter import _Padding

import numpy as np
import mindspore
from mindspore import ops, Tensor
from ..auto_class import AutoProcessor, AutoModel
from ..mindformer_book import MindFormerBook
from .base_pipeline import BasePipeline
from ..tools.register import MindFormerRegister, MindFormerModuleType
from ..models import BaseModel, Tokenizer

__all__ = ['FillMaskPipeline']

@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="fill_mask")
class FillMaskPipeline(BasePipeline):
    """
    Pipeline for mask fill

    Args:
        model: a pretrained model (str or BaseModel) in _supproted_list.
        tokenizer : a tokenizer (None or Tokenizer) for text processing
    """
    _support_list = MindFormerBook.get_model_support_list()['bert']
    return_name = 'fillmask'

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
            raise TypeError(f"model should be inherited from BaseModel, but got type {type(model)}.")

        if tokenizer is None:
            raise ValueError(f"{self.__class__.__name__}"
                             " requires for a tokenizer.")
        self.input_text = ""

        super().__init__(model, tokenizer, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        """sanitize parameters for preprocess, forward, and postprocess."""
        if 'batch_size' in pipeline_parameters:
            raise ValueError(f"The {self.__class__.__name__} does not support batch inference, please remove the "
                             f"batch_size")

        postprocess_params = {}

        forward_key_name = ['top_k', 'top_p', 'do_sample', 'eos_token_id', 'repetition_penalty', 'max_length']
        forward_kwargs = {}
        for item in forward_key_name:
            if item in pipeline_parameters:
                forward_kwargs[item] = pipeline_parameters.get(item)

        preprocess_key_name = ['top_k', 'top_p', 'do_sample', 'eos_token_id', 'repetition_penalty', 'max_length',
                               'padding']
        preprocess_params = {k: v for k, v in pipeline_parameters.items() if k in preprocess_key_name}
        return preprocess_params, forward_kwargs, postprocess_params

    def preprocess(self, inputs, **preprocess_params):
        """
        Preprocess of mask fill

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
        self.input_text = inputs if isinstance(inputs, str) else ""
        max_length = preprocess_params.pop("max_length", 128)
        padding = preprocess_params.pop("padding", "max_length")
        inputs = self.tokenizer(inputs, max_length=max_length, padding=padding,
                                return_tensors="ms", **preprocess_params)
        expand_dims = ops.ExpandDims()
        return {"input_ids": expand_dims(inputs["input_ids"], 0),
                "input_mask": expand_dims(inputs["attention_mask"], 0),
                "token_type_id": expand_dims(inputs["token_type_ids"], 0),
                "masked_lm_positions": expand_dims(Tensor(self.tokenizer.mask_index), 0)}

    def forward(self, model_inputs, **forward_params):
        """
        Forward process

        Args:
            model_inputs (dict): outputs of preprocess.

        Return:
            probs dict.
        """
        forward_params.pop("None", None)
        output_ids = self.model(**model_inputs)
        return {"output_ids": output_ids}

    def postprocess(self, model_outputs, **postprocess_params):
        """
        Postprocess

        Args:
            model_outputs (dict): outputs of forward process.

        Return:
            The generated results
        """
        outputs = model_outputs["output_ids"][-2].asnumpy()
        tokens_dict = []
        max_tokens = np.argmax(outputs, axis=1)
        for ind, tokenid in enumerate(max_tokens):
            token = self.tokenizer.decode([int(tokenid),], skip_special_tokens=True)
            token = token.replace(' ', '')
            tokens_dict.append({'score': outputs[ind, tokenid],
                                'token': tokenid,
                                'token_str': token})
            self.input_text = self.input_text.replace('[MASK]', token, 1)
        tokens_dict.append({'sequence': self.input_text})
        return [tokens_dict,]
