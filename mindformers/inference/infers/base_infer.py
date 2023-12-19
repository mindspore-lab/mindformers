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

from mindformers.inference.context import build_context
from mindformers.inference.infer_config import InferConfig
from mindformers.models import BaseTokenizer, BaseImageProcessor
from mindformers.tools.logger import logger


class DynShapeGear:
    """
    Dynamical shape gear setting
    """

    def __init__(self, gear_config_path):
        self.seq_gears, self.bs_gears = DynShapeGear.parse_dynamic_gears(gear_config_path)

    def match_seq(self, input_seq_length):
        """match seq length gear"""
        if input_seq_length < 0:
            err_msg = f"Match seq length gear failed, input length({input_seq_length}) must be not less than 0."
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        if input_seq_length > self.seq_gears[-1]:
            return self.seq_gears[-1]
        gear = list(filter(lambda k: k >= input_seq_length, self.seq_gears))[0]
        logger.info(f"Match seq length gear: {gear}")
        return gear

    def match_bs(self, input_batch_size):
        """match batch size gear"""
        if input_batch_size < 0:
            err_msg = f"Match batch size gear failed, input length({input_batch_size}) must be not less than 0."
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        if input_batch_size > self.bs_gears[-1]:
            return self.bs_gears[-1]
        gear = list(filter(lambda k: k >= input_batch_size, self.bs_gears))[0]
        logger.info(f"Match batch size gear: {gear}")
        return gear

    @classmethod
    def parse_dynamic_gears(cls, ge_config_path):
        """
        Parse full model ge config to get input shape gear list.
        Input config format likes this:
            'ge.inputShape=batch_index:-1;batch_valid_length:-1;input_position:-1;tokens:-1,1;zactivate_len:-1
            'ge.dynamicDims=1,1,1,1,1024;8,8,8,8,1024;8,8,8,8,4096;'

        :param ge_graph_options: The path of lite dynamical dims config file.
        :return: gear list
        """
        dims = cls.get_dynamic_shape_str(ge_config_path)
        if len(dims) < 2:
            return "pure dynamic"
        seq_gears, bs_gears = [], []
        key = [k.split(':') for k in dims[0].split(';')]
        val = [n.split(',') for n in dims[1].split(';') if n != '']
        for v in val:
            if len(key) != len(v):
                err_msg = f"ge.inputShape and ge.dynamicDims not match"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

        for i in range(len(key)):
            if key[i][0] == 'tokens' and '-1' in key[i][1]:
                bs_gears.extend([int(n[i]) for n in val])
            if key[i][0] == 'zactivate_len' and '-1' in key[i][1]:
                seq_gears.extend([int(n[i]) for n in val])
        seq_gears.sort()
        bs_gears.sort()
        return seq_gears, bs_gears

    @classmethod
    def get_dynamic_shape_str(cls, ge_config_path):
        """get gear info from .ini file"""
        res = []
        with open(ge_config_path, 'r') as f:
            for line in f:
                # get key
                if "ge.inputShape" in line and line[0] not in ['#', ';']:
                    res.append(line.split('=')[1].split('\n')[0])
                # get val
                if "ge.dynamicDims" in line and line[0] not in ['#', ';']:
                    res.append(line.split('=')[1].split('\n')[0])
        if not res:
            logger.warning("ge config file %s has no item named ge.dynamicDims", ge_config_path)
            return None
        return res


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
        self.dynamic = config.dynamic

        self.context = build_context(config=self.config)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.full_model = None
        self.cache_model = None
        if config.prefill_model_path and config.increment_model_path:
            if isinstance(self.config_path, (list, tuple)) and len(self.config_path) > 1:
                self.prefill_config = self.config_path[0]
                self.increment_config = self.config_path[1]
            else:
                self.prefill_config = self.config_path
                self.increment_config = self.config_path
            self.full_model, self.cache_model = self._load_increment_models(
                config.prefill_model_path, config.increment_model_path, self.prefill_config, self.increment_config
            )
        else:
            self.full_model = self._load_model(config.prefill_model_path)
        if self.dynamic:
            self.dynshape_gears = DynShapeGear(self.increment_config)

    def _load_model(self, model_path):
        """ load single model from model path.
        """
        model = Model()
        model.build_from_file(model_path, model_type=self.model_type,
                              context=self.context, config_path=self.config_path)
        return model

    def _load_increment_models(self, full_model_path, cache_model_path, prefill_config, increment_config):
        """
        load kv cache models.
        """
        full_model = Model()
        cache_model = Model()

        model_group = ModelGroup(ModelGroupFlag.SHARE_WEIGHT)
        model_group.add_model([full_model, cache_model])

        full_model.build_from_file(full_model_path, self.model_type, self.context, prefill_config)
        cache_model.build_from_file(cache_model_path, self.model_type, self.context, increment_config)

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
