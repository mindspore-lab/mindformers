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
"""base infer class"""

import abc
from typing import Optional

from mindspore_lite import Model, ModelGroup, ModelGroupFlag

from mindformers.models import BaseTokenizer, BaseImageProcessor
from mindformers.inference.infer_config import InferConfig
from mindformers.inference.context import build_context


class BaseInfer(metaclass=abc.ABCMeta):
    """
    BaseInfer.
    """
    def __init__(self,
                 config: InferConfig = None,
                 tokenizer: Optional[BaseTokenizer] = None,
                 image_processor: Optional[BaseImageProcessor] = None):
        if config is None:
            config = InferConfig()
        self.config = config
        self.model_type = config.model_type
        self.model_name = config.model_name
        self.seq_length = config.seq_length
        self.config_path = config.config_path
        self.context = build_context(config=self.config)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.full_model = None
        self.cache_model = None
        if config.prefill_model_path and config.increment_model_path:
            self.full_model, self.cache_model = self._load_increment_models(
                config.prefill_model_path, config.increment_model_path
            )
        else:
            self.full_model = self._load_model(config.prefill_model_path)

    def _load_model(self, model_path):
        """ load single model from model path.
        """
        model = Model()
        model.build_from_file(model_path, model_type=self.model_type,
                              context=self.context, config_path=self.config_path)
        return model

    def _load_increment_models(self, full_model_path, cache_model_path):
        """
        load kv cache models.
        """
        full_model = Model()
        cache_model = Model()

        model_group = ModelGroup(ModelGroupFlag.SHARE_WEIGHT)
        model_group.add_model([full_model, cache_model])

        full_model.build_from_file(full_model_path, self.model_type, self.context, self.config_path)
        cache_model.build_from_file(cache_model_path, self.model_type, self.context, self.config_path)

        return full_model, cache_model

    def __call__(self, inputs, **kwargs):
        """call infer process."""
        return self.infer(inputs, **kwargs)

    @abc.abstractmethod
    def infer(self, inputs, **kwargs):
        """infer interface."""

    @abc.abstractmethod
    def preprocess(self, input_data, **kwargs):
        """preprocess interface."""

    @abc.abstractmethod
    def postprocess(self, predict_data, **kwargs):
        """postprocess interface."""
