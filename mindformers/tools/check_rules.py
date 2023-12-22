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
"""Functions to check rules"""
import os
import json
import mindspore as ms
from .utils import get_real_group_size
from .logger import logger


def get_parallel_strategy(config):
    dp = config.parallel_config.data_parallel
    mp = config.parallel_config.model_parallel
    pp = config.parallel_config.pipeline_stage
    return dp, mp, pp


def get_device_num():
    return get_real_group_size()


def get_server_num():
    path = os.getenv('RANK_TABLE_FILE', None)
    if path is None:
        return 1
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return int(data['server_count'])


def _check_mode(config, mode):
    """rules with different mode"""
    if mode == 'train':
        if config.model.model_config.use_past:
            config.model.model_config.use_past = False
            logger.warning("use_past could not be used in train mode, "
                           "it has been forced to False")
    elif mode == 'predict':
        pass
    elif mode == 'eval':
        pass
    else:
        raise ValueError(f"mode should be in ['train', 'predict', 'eval'], but get {mode}")


def _check_full_batch():
    """check full_batch"""
    parallel_mode = ms.get_auto_parallel_context("parallel_mode")
    full_batch = ms.get_auto_parallel_context("full_batch")
    if parallel_mode not in ["semi_auto_parallel", "auto_parallel"] and full_batch:
        ms.set_auto_parallel_context(full_batch=False)
        logger.warning(f"full_batch could only be used under semi_auto_parallel or auto_parallel, "
                       f"but get {parallel_mode}, full_batch has been forced to False")


def _check_parallel(config):
    """check parallel config"""
    parallel_mode = ms.get_auto_parallel_context("parallel_mode")
    dp, mp, pp = get_parallel_strategy(config)
    device_num = get_device_num()
    server_num = get_server_num()
    if parallel_mode in ["semi_auto_parallel"]:
        if dp * mp * pp != device_num:
            raise ValueError(f"The parallel config data_parallel * model_parallel * pipeline_stage should "
                             f"be equal to device_num, but get dp*mp*pp = {dp}*{mp}*{pp} = {dp * mp * pp} "
                             f"!= device_num({device_num})")

        if config.model.model_config.num_layers and config.model.model_config.num_layers < pp:
            raise ValueError(f"num_layers of model should be greater than or equal to pipeline_stage, but get "
                             f"num_layers ({config.model.model_config.num_layers}) < pp({pp})")

        if server_num > 1:
            if server_num % pp != 0:
                logger.warning(f"server_num % pipeline_stage = {server_num} % {pp} = {server_num % pp} != 0, "
                               f"which may cause parallel error when using multiple servers")

        if config.parallel.enable_parallel_optimizer:
            if config.model.model_config.vocab_size and config.model.model_config.vocab_size % device_num != 0:
                logger.warning(f"vocab_size({config.model.model_config.vocab_size}) % device_num({device_num})"
                               f" = {config.model.model_config.vocab_size % device_num} != 0, which "
                               f"may cause the optimizer parallel of the relevant parameters to fail")
            if config.model.model_config.hidden_size and config.model.model_config.hidden_size % device_num != 0:
                logger.warning(f"hidden_size({config.model.model_config.hidden_size}) % device_num({device_num})"
                               f" = {config.model.model_config.hidden_size % device_num} != 0, which "
                               f"may cause the optimizer parallel of the relevant parameters to fail")


def check_rules(config, mode='train'):
    """check rules"""
    _check_mode(config, mode)
    _check_full_batch()
    _check_parallel(config)
