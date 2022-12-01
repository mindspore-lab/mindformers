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

'''BasePipeline'''
from abc import ABC, abstractmethod

from tqdm import tqdm
from mindspore.dataset import (
    GeneratorDataset, VisionBaseDataset,
    SourceDataset, MappableDataset
)
from mindformers.tools import logger
from mindformers.mindformer_book import print_dict

class BasePipeline(ABC):
    '''Base pipeline'''
    _support_list = {}

    def __init__(self, model, tokenizer=None, feature_extractor=None, **kwargs):
        super(BasePipeline, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self._preprocess_params, self._forward_params,\
        self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0

        self._batch_size = kwargs.pop("batch_size", None)

    def __call__(self, inputs, batch_size=None, **kwargs):
        '''
        Call method

        Args:
            inputs (Dataset, list, etc): the inputs of pipeline, the type of input depends on task.

        Return:
            outputs, depends on task
        '''
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        is_dataset = isinstance(inputs, (
            GeneratorDataset, VisionBaseDataset,
            MappableDataset, SourceDataset))
        is_list = isinstance(inputs, list)

        self.call_count += 1
        if self.call_count > 10 and not is_dataset:
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
        '''sanitize parameters for preprocess, forward and postprocess.'''
        raise NotImplementedError("_sanitize_parameters not implemented")

    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        '''run a single input'''
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def run_multi(self, inputs, preprocess_params, forward_params, postprocess_params):
        '''run multiple input'''
        outputs = []
        for item in inputs:
            outputs.extend(self.run_single(item, preprocess_params,
                                           forward_params, postprocess_params))
        return outputs

    @abstractmethod
    def preprocess(self, input_, **preprocess_parameters):
        '''preprocess'''
        raise NotImplementedError("preprocess not implemented.")

    @abstractmethod
    def forward(self, model_inputs, **forward_params):
        '''forward'''
        raise NotImplementedError("forward not implemented.")

    @abstractmethod
    def postprocess(self, model_outputs, **postprocess_parameters):
        '''postprocess'''
        raise NotImplementedError("postprocess not implemented.")

    @classmethod
    def show_support_list(cls):
        '''show_support_list'''
        logger.info("support list of %s is:", cls.__name__)
        print_dict(cls._support_list)
