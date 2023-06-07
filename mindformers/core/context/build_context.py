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
"""Build context."""

import os

from mindspore import context
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs

from mindformers.core.callback import ProfileMonitor
from mindformers.trainer.config_args import ContextConfig, ParallelContextConfig
from mindformers.tools.register import MindFormerConfig
from mindformers.tools import PARALLEL_MODE, MODE, get_output_subpath
from mindformers.tools.logger import logger


CONTEXT_CONFIG = {
    'mode': 'GRAPH_MODE', 'device_target': 'Ascend', 'device_id': 0, 'save_graphs': False}
PARALLEL_CONFIG = {'parallel_mode': 'DATA_PARALLEL', 'gradients_mean': True}


def build_context(config):
    """Build context."""
    if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
        config = MindFormerConfig(**config)
    if config.use_parallel and config.parallel_config.pipeline_stage > 1:
        config.parallel.pipeline_stages = config.parallel_config.pipeline_stage
    local_rank, device_num = init_context(use_parallel=config.use_parallel,
                                          context_config=config.context, parallel_config=config.parallel)
    set_algo_parameters(elementwise_op_strategy_follow=True, fully_use_devices=True)
    _set_multi_subgraphs()

    config.device_num = device_num
    config.local_rank = local_rank

    if config.parallel.get("strategy_ckpt_load_file"):
        context.set_auto_parallel_context(strategy_ckpt_load_file=config.parallel.strategy_ckpt_load_file)


def init_context(use_parallel=True, context_config=None, parallel_config=None):
    """Context initialization for MindSpore."""

    if isinstance(context_config, ContextConfig):
        context_config = context_config.__dict__
    if isinstance(parallel_config, ParallelContextConfig):
        parallel_config = parallel_config.__dict__

    if context_config is None:
        context_config = CONTEXT_CONFIG
    if parallel_config is None:
        parallel_config = PARALLEL_CONFIG

    _set_check_context_config(context_config)
    _set_check_parallel_config(parallel_config)

    device_num = 1
    rank_id = 0
    context_config['mode'] = MODE.get(context_config.get('mode'))

    context.set_context(max_device_memory=context_config.get('max_device_memory'),
                        mode=context_config.get('mode'))
    del context_config['mode']
    del context_config['max_device_memory']
    if use_parallel:
        init()
        device_id = int(os.getenv('DEVICE_ID', '0'))  # 0 ~ 7
        rank_id = get_rank()  # local_rank
        device_num = get_group_size()  # world_size
        context_config['device_id'] = device_id
        parallel_config['parallel_mode'] = PARALLEL_MODE.get(parallel_config.get('parallel_mode'))
        parallel_config.setdefault('device_num', device_num)
        context.set_context(**context_config)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(**parallel_config)
    else:
        context.set_context(**context_config)
    return rank_id, device_num


def build_profile_cb(config):
    """build profile callback from config."""
    start_profile = config.init_start_profile
    profile_communication = config.profile_communication
    if config.device_num > 1:
        logger.info("Device number is %s > 1, so profile_communication and start_profile will be set True ")
        start_profile = True
        profile_communication = True
    profile_cb = ProfileMonitor(
        start_step=config.profile_start_step,
        stop_step=config.profile_stop_step,
        start_profile=start_profile,
        profile_communication=profile_communication,
        profile_memory=config.profile_memory)
    logger.warning(
        "In profiler mode, data sink mode will be turned off. "
        "Please reduce the data sample size with 'num_samples' in MindSpore data format according to "
        "https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling_ascend.html.")
    logger.warning("In profiler mode, auto-tune will be turned off.")
    config.runner_config.sink_mode = False
    config.auto_tune = False

    return profile_cb


def _set_check_context_config(config):
    """Set context config."""
    mode = config.get('mode')
    if mode is None:
        config.setdefault('mode', 0)
    if mode not in MODE.keys():
        raise IndexError('Running mode should be in {}, but get {}'.format(MODE.keys(), mode))

    device = config.get('device_id')
    if device is None:
        config.setdefault('device_id', 0)

    max_device_memory = config.get('max_device_memory')
    if max_device_memory is None:
        config.setdefault('max_device_memory', '30GB')

    save_graph = config.get('save_graphs')
    if save_graph:
        save_graphs_path = config.get('save_graphs_path')
        if save_graphs_path is None:
            save_graphs_path = get_output_subpath("debug/graphs_info", append_rank=False)
        config.setdefault('save_graphs_path', save_graphs_path)
    enable_dump = config.get('enable_dump')
    if enable_dump:
        save_dump_path = config.get('save_dump_path')
        if save_dump_path is None:
            save_dump_path = get_output_subpath("debug/dump_info", append_rank=False)
        config.setdefault('save_dump_path', save_dump_path)


def _set_check_parallel_config(config):
    """Set parallel config."""
    parallel_mode = config.get('parallel_mode')
    if parallel_mode is None:
        config.setdefault('parallel_mode', 0)

    if PARALLEL_MODE.get(config.get('parallel_mode')) not in \
            [context.ParallelMode.SEMI_AUTO_PARALLEL, context.ParallelMode.AUTO_PARALLEL] and config.get('full_batch'):
        logger.info("full_batch will be forced to False when the parallel mode is stand_alone or data_parallel")
        config.setdefault('full_batch', False)

    if parallel_mode not in PARALLEL_MODE.keys():
        raise IndexError(
            'Running parallel mode should be in {}, but get {}'.format(PARALLEL_MODE.keys(), parallel_mode))
