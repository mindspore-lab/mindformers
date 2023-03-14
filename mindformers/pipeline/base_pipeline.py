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
# This file was refer to project:
# https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py
# ============================================================================

"""BasePipeline"""
from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np

from tqdm import tqdm
from mindspore import Tensor
from mindspore.dataset import (
    GeneratorDataset, VisionBaseDataset,
    SourceDataset, MappableDataset
)
from mindspore.dataset.engine.datasets import BatchDataset, RepeatDataset

from mindformers.tools import logger
from mindformers.mindformer_book import print_dict
from ..models import BaseModel, BaseTokenizer, BaseImageProcessor


class BasePipeline(ABC):
    r"""
    Base Pipeline For All Task Pipelines

    Args:
        model (Union[str, BaseModel]): The model used to perform task,
            the input could be a supported model name, or a model instance
            inherited from BaseModel.
        tokenizer (Optional[BaseTokenizer]): The tokenizer of model, it could be
            None if the model do not need tokenizer.
        image_processor (Optional[BaseImageProcessor]): The image_processor of model,
            it could be None if the model do not need image_processor.
    """
    _support_list = {}

    def __init__(self, model: Union[str, BaseModel],
                 tokenizer: Optional[BaseTokenizer] = None,
                 image_processor: Optional[BaseImageProcessor] = None,
                 **kwargs):
        super(BasePipeline, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self._preprocess_params, self._forward_params, \
        self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0

        self._batch_size = kwargs.pop("batch_size", None)

    def __call__(self, inputs: Union[GeneratorDataset, list, str, Tensor, np.array],
                 batch_size: Optional[int] = None,
                 **kwargs):
        r"""Call Method

        Args:
            inputs (Union[GeneratorDataset, list, str, etc]): The inputs of pipeline,
                the type of inputs depends on task.
            batch_size (Optional[int]): The batch size for a GeneratorDataset input, for
                other types of inputs, the batch size would be set to 1 by default.
        """
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        is_dataset = isinstance(inputs, (
            GeneratorDataset, VisionBaseDataset,
            MappableDataset, SourceDataset))
        is_list = isinstance(inputs, list)

        self.call_count += 1
        if self.call_count == 20 and not is_dataset:
            logger.info("You seem to be using the pipeline sequentially for"
                        " numerous samples. In order to maximize efficiency"
                        " please set input a mindspore.dataset.GeneratorDataset.")

        if batch_size is None:
            if self._batch_size is None:
                batch_size = 1
            else:
                batch_size = self._batch_size

        if batch_size > 1 and not is_dataset:
            batch_size = 1
            logger.info("batch_size is set to 1 for non-dataset inputs.")

        if is_dataset:
            logger.info("dataset is processing.")
            if not isinstance(inputs, (BatchDataset, RepeatDataset)):
                inputs = inputs.batch(batch_size)

            outputs = []
            for items in tqdm(inputs.create_dict_iterator()):
                outputs.extend(self.run_single(items, preprocess_params,
                                               forward_params, postprocess_params))
        elif is_list:
            outputs = self.run_multi(inputs, preprocess_params, forward_params, postprocess_params)
        else:
            outputs = self.run_single(inputs, preprocess_params, forward_params, postprocess_params)

        return outputs

    @abstractmethod
    def _sanitize_parameters(self, **pipeline_parameters):
        r"""Sanitize Parameters

        Args:
            pipeline_parameters (Optional[dict]): The parameter dict to be parsed.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("_sanitize_parameters not implemented")

    def run_single(self, inputs: Union[dict, str, np.array, Tensor],
                   preprocess_params: dict,
                   forward_params: dict,
                   postprocess_params: dict):
        r"""Run Single method
        This function is used to run a single forward process for task.

        Args:
            inputs (Union[dict, str, etc]): The inputs of pipeline,
                the type of inputs depends on task.
            preprocess_params (dict): The parameter dict for preprocess.
            forward_params (dict): The parameter dict for model forward process.
            postprocess_params (dict): The parameter dict for postprocess.
        """
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def run_multi(self, inputs: Union[list, tuple],
                  preprocess_params: dict,
                  forward_params: dict,
                  postprocess_params: dict):
        r"""Run Multiple Method
        This function is used to run a list input for task.

        Args:
            inputs (Union[list, tuple, iterator]): The iterable input for pipeline.
            preprocess_params (dict): The parameter dict for preprocess.
            forward_params (dict): The parameter dict for model forward process.
            postprocess_params (dict): The parameter dict for postprocess.
        """
        outputs = []
        for item in inputs:
            outputs.extend(self.run_single(item, preprocess_params,
                                           forward_params, postprocess_params))
        return outputs

    @abstractmethod
    def preprocess(self, inputs: Union[dict, str, np.array, Tensor],
                   **preprocess_params):
        r"""The Preprocess For Task

        Args:
            inputs (Union[dict, str, etc]): The inputs of pipeline,
                the type of inputs depends on task.
            preprocess_params (dict): The parameter dict for preprocess.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("preprocess not implemented.")

    @abstractmethod
    def forward(self, model_inputs: Union[dict, str, np.array, Tensor],
                **forward_params):
        r"""The Forward Process of Model

        Args:
            model_inputs (Union[dict, str, etc]): The output of preprocess,
                the type of model_inputs depends on task.
            forward_params (dict): The parameter dict for model forward.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("forward not implemented.")

    @abstractmethod
    def postprocess(self, model_outputs: Union[dict, str, np.array, Tensor],
                    **postprocess_params):
        r"""The Postprocess of Task

        Args:
            model_outputs (Union[dict, str, etc]): The output of model forward,
                the type of model_outputs depends on task.
            postprocess_params (dict): The parameter dict for post process.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("postprocess not implemented.")

    @classmethod
    def show_support_list(cls):
        """show_support_list"""
        logger.info("support list of %s is:", cls.__name__)
        print_dict(cls._support_list)
