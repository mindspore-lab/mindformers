# Copyright 2024 Huawei Technologies Co., Ltd
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
from dataclasses import dataclass
from typing import Union

import mindspore as ms
import mindspore.dataset as ds
import mindspore.context as ms_context
import psutil

from mindformers.core.context.parallel import ParallelOperator
from mindformers.core.context.validators import RunMode, execute_validator
from mindformers.tools import MODE, get_output_subpath
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.utils import check_in_dynamic_cluster
from mindformers.trainer.config_args import (
    ContextConfig,
    MFContextConfig,
    ParallelContextConfig,
)
from mindformers.trainer.training_args import TrainingArguments
from mindformers.utils import get_cann_workqueue_cores


class Context:
    """The wrapper of mindformer context and mindspore context."""

    # pylint: disable=W0613
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Context, cls).__new__(cls)
        return cls._instance

    def __init__(self, config=None):
        """Build context."""
        if not hasattr(self, '_initailed'):
            self.rank_id = 0
            self.device_num = 1
            self.config = config if config is not None else MindFormerConfig()
            self.mf_ctx_opr = MFContextOperator(self.config)
            self.ms_ctx_opr = MSContextOperator(self.config)
            self.parallel_opr = ParallelOperator(self.config)

            if self.config.use_parallel:
                self.rank_id, self.device_num = (
                    self.parallel_opr.init_communication()
                )
            set_cpu_affinity(self.rank_id, self.device_num)

            self._initailed = True

    def set_mf_ctx_run_mode(self, run_mode):
        if run_mode is not None:
            members = [k.lower() for k, _ in RunMode.__members__.items()]
            if run_mode in members:
                self.mf_ctx_opr.run_mode = run_mode
            else:
                raise ValueError(
                    f'Fail to set run_mode, Invalid value. '
                    f'Expected one of {members}, got {run_mode}'
                )


@dataclass
class MSContextOperator:
    """The wrapper of mindspore context operation."""

    def __init__(self, config):
        self.config = config
        ms_kwargs = self._handle_data()
        logger.debug('MSContextConfig load configs: %s', ms_kwargs)
        self.set_context(**ms_kwargs)
        del self.config

    def _handle_data(self):
        """Get the valid ms config."""
        ctx = self.config.get('context', {})
        ms_ctx = {
            'device_id': ctx.get('device_id', 0),
            'max_device_memory': ctx.get('max_device_memory', '1024GB'),
            'mode': MODE.get(ctx.get('mode', 'GRAPH_MODE')),
            'deterministic': "ON" if ctx.get('train_precision_sync') else "OFF"
        }
        self._set_device_id(ctx, ms_ctx)
        self._set_save_graphs_path(ctx, ms_ctx)
        self._set_save_dump_path(ctx, ms_ctx)
        self._set_jit_config(ctx, ms_ctx)
        self._set_runtime_num_threads(ctx, ms_ctx)
        return self._remove_mf_keys({**ctx, **ms_ctx})

    def _set_device_id(self, ctx, ms_ctx):
        if self.config.use_parallel and check_in_dynamic_cluster():
            # for dynamic cluster, we should not set device id in context.
            ctx.pop('device_id', None)
        else:
            ms_ctx['device_id'] = int(os.getenv('DEVICE_ID', '0'))

    def _set_save_graphs_path(self, ctx, ms_ctx):
        if ctx.get('save_graphs'):
            ms_ctx['save_graphs_path'] = ctx.get(
                'save_graphs_path',
                get_output_subpath("debug/graphs_info", append_rank=False)
            )

    def _set_save_dump_path(self, ctx, ms_ctx):
        if ctx.get('enable_dump'):
            ms_ctx['save_dump_path'] = ctx.get(
                'save_dump_path',
                get_output_subpath("debug/dump_info", append_rank=False)
            )

    def _set_jit_config(self, ctx, ms_ctx):
        """Get jit_level and infer_boost from config and set into ms context."""
        run_mode = self.config.get('run_mode')
        use_past = MindFormerConfig.get_nested_config(
            self.config, ['model', 'model_config', 'use_past'], False
        )

        if (
                run_mode is not None
                and RunMode(run_mode) in [RunMode.PREDICT, RunMode.EVAL]
                and use_past
        ):
            jit_level = ctx.get("jit_level", "O0")
            infer_boost = ctx.get("infer_boost", "on")
            jit_config = ctx.get("jit_config")
            if jit_config is not None:
                jit_level = jit_config.get("jit_level") or jit_level
                infer_boost = jit_config.get("infer_boost") or infer_boost
            logger.info(
                "Predict context config, jit_level: "
                f"{jit_level}, infer_boost: {infer_boost}"
            )

            if jit_level == "O1":
                raise ValueError(
                    "jit_level O1 is not supported in predict mode."
                )
            if jit_level == "O2" and infer_boost == "on":
                raise ValueError(
                    "Only infer_boost must set off to "
                    "set jit_level to O2 in predict mode."
                )

            ms_ctx['jit_config'] = {
                "jit_level": jit_level,
                "infer_boost": infer_boost
            }

    def _set_runtime_num_threads(self, ctx, ms_ctx):
        ms_version = ms.__version__
        if ctx.get('runtime_num_threads') is None and ms_version < "2.3.0":
            ms_ctx['runtime_num_threads'] = 1
            logger.info(
                "The current MindSpore version is %s,"
                "and set the default runtime_num_threads to 1.", ms_version
            )

    def _remove_mf_keys(self, ctx_config):
        mf_keys = MFContextConfig.get_supported_kwargs()
        return {k: v for k, v in ctx_config.items() if k not in mf_keys}

    def set_context(self, **kwargs):
        """Set ms context value according to the input key words."""
        ms_context.set_context(**kwargs)

    def get_context(self, attr_key):
        """Get ms context attribute value according to the input key."""
        return ms_context.get_context(attr_key)


@dataclass
class MFContextOperator(MFContextConfig):
    """The wrapper of mindformers context operation."""

    def __init__(self, config):
        self.config = config
        supported_kwargs = self._handle_data()
        logger.debug('MFContextConfig load configs: %s', supported_kwargs)
        super(MFContextOperator, self).__init__(**supported_kwargs)
        use_past = MindFormerConfig.get_nested_config(
            self.config, ['model_config', 'use_past'], False
        )
        self.set_env(use_past)
        del self.config

    def _handle_data(self):
        ctx_config = self.config.get('context', {})
        extra_config = self._get_first_level_config()
        return self.filter_kwargs(**{**extra_config, **ctx_config})

    def _get_first_level_config(self):
        return {
            k: v
            for k, v in self.config.items()
            if k in MFContextOperator.get_supported_kwargs()
        }

    def set_env(self, use_past=False):
        """Update environment variables."""
        if self.train_precision_sync:
            hccl_deterministic = 'true'
            ascend_launch_blocking = '1'
            te_parallel_compiler = '1'
        else:
            hccl_deterministic = 'false'
            ascend_launch_blocking = '0'
            te_parallel_compiler = '0'

        if self.infer_precision_sync:
            custom_matmul_shuffle = 'off'
            lccl_deterministic = '1'
        else:
            custom_matmul_shuffle = 'on'
            lccl_deterministic = '0'

        env = {
            'HCCL_DETERMINISTIC': hccl_deterministic,
            'ASCEND_LAUNCH_BLOCKING': ascend_launch_blocking,
            'TE_PARALLEL_COMPILER': te_parallel_compiler,
            'CUSTOM_MATMUL_SHUFFLE': custom_matmul_shuffle,
            'LCCL_DETERMINISTIC': lccl_deterministic,
            'MS_ENABLE_GRACEFUL_EXIT': '1' if self.use_graceful_exit else '0'
        }

        run_mode = (
            getattr(self, 'run_mode') if hasattr(self, 'run_mode') else None
        )
        if (
                run_mode is not None
                and RunMode(run_mode) in [RunMode.PREDICT, RunMode.EVAL]
                and use_past
        ):
            env['MS_ALLOC_CONF'] = 'enable_vmm:False'
            env['RUN_MODE'] = run_mode
            env['CPU_AFFINITY'] = 'True'
            env['MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST'] = 'PagedAttention'

        if (
                self.enable_mindio_ttp_save_ckpt and
                self.config.runner_config.sink_size == 1
        ):
            os.environ['MS_ENABLE_TFT'] = '{TTP:1 UCE:1}'
            os.environ['MINDIO_FOR_MINDSPORE'] = '1'

        os.environ.update(env)

    def set_context(self, **kwargs):
        """Set mf context value according to the input key words."""
        supported_kwargs = self.filter_kwargs(**kwargs)
        for k, v in supported_kwargs.items():
            setattr(self, k, v)

    def get_context(self, attr_key):
        """Set mf context value according to the input key words."""
        return getattr(self, attr_key, None)


def set_cpu_affinity(rank_id, rank_size):
    """cpu affinity"""
    use_cpu_affinity = os.environ.get('CPU_AFFINITY')
    if use_cpu_affinity and use_cpu_affinity.lower() in ('1', 'true'):
        ds.config.set_numa_enable(True)
        count = psutil.cpu_count()
        current_process = psutil.Process()
        used_cpus_num = count // rank_size
        used_cpus = list(
            range(rank_id * used_cpus_num, (rank_id + 1) * used_cpus_num)
        )
        cann_used_cpus = get_cann_workqueue_cores(rank_id)
        logger.info(f"cann workqueue cpus: {cann_used_cpus}")
        used_cpus = list(set(used_cpus) - set(cann_used_cpus))
        if not used_cpus:
            # cann setup all cpus, disable the binding cores
            logger.warning(
                f"CANN use cpus: {cann_used_cpus}, "
                "model get empty cpu list, disable binding cores"
            )
            used_cpus = list(
                range(rank_id * used_cpus_num, (rank_id + 1) * used_cpus_num)
            )
        current_process.cpu_affinity(used_cpus)
        logger.info(
            f"cpu_affinity, rank_id: {rank_id}, device_num: {rank_size}"
        )


def init_context(
        use_parallel=False, context_config=None, parallel_config=None
):
    """
    Initialize the context.

    Args:
        use_parallel (bool): Whether to use parallel, default: False.
        context_config (Union[dict, ContextConfig]):
        The context config, default: None.
        parallel_config (Union[dict, ParallelContextConfig]):
        The parallel context config, default: None.

    Returns:
        - Int, the local_rank number.
        - Int, the total available devices number.

    Examples:
        >>> from mindformers import init_context
        >>> init_context(use_parallel=False)
    """

    if isinstance(context_config, ContextConfig):
        context_config = context_config.__dict__

    if isinstance(parallel_config, ParallelContextConfig):
        parallel_config = parallel_config.__dict__

    config = {
        'use_parallel': use_parallel,
        'context': context_config,
        'parallel': parallel_config
    }
    ctx = build_context(config)
    return ctx.rank_id, ctx.device_num


def build_context(config: Union[dict, MindFormerConfig, TrainingArguments]):
    """
    Build the context from config.

    Note:
        parameter config must contain keys: 'context', 'parallel',
        when config is dict.

    Args:
        config (Union[dict, MindFormerConfig, TrainingArguments]):
        The configuration to initialize the context.
        This can be a dictionary, a MindFormerConfig instance,
        or a TrainingArguments instance.

    Returns:
        _Context, The instantiated context.

    Examples:
        >>> from mindformers import build_context
        >>> config = {'context': {'mode': 'GRAPH_MODE'}, 'parallel':{}}
        >>> build_context(config=config)
    """
    if isinstance(config, TrainingArguments):
        config = config.convert_args_to_mindformers_config()

    config['parallel_config'] = config.get('parallel_config', {})
    config = MindFormerConfig(**config)

    execute_validator(config)

    return Context(config)


def set_context(run_mode=None, **kwargs):
    """
    Set context for running environment.

    Context should be configured before running your program.
    If there is no configuration,
    it will be automatically set according to the device target by default.

    Note:
        Attribute name is required for setting attributes.
        Currently only run_mode belongs to MindFormers context.
        The kwargs will be passed to MindSpore set_context.

    Args:
        run_mode (str): The mode of the model behaviour.
        Must be in ['train', 'finetune', 'eval', 'predict'].
        **kwargs: MindSpore context arguments.

    Examples:
        >>> from mindformers import build_context, set_context
        >>> config = {'context': {'mode': 'GRAPH_MODE'}, 'parallel':{}}
        >>> build_context(config=config)
        >>> set_context(max_device_memory='59GB')
    """
    ctx = Context()
    ctx.set_mf_ctx_run_mode(run_mode)

    train_precision_sync = kwargs.pop('train_precision_sync', None)
    infer_precision_sync = kwargs.pop('infer_precision_sync', None)
    if train_precision_sync is not None:
        ctx.mf_ctx_opr.train_precision_sync = train_precision_sync
        deterministic = "ON" if train_precision_sync else "OFF"
        ctx.ms_ctx_opr.set_context(deterministic=deterministic)
        ctx.mf_ctx_opr.set_env()

    if infer_precision_sync is not None:
        ctx.mf_ctx_opr.infer_precision_sync = infer_precision_sync
        ctx.mf_ctx_opr.set_env()

    mf_ctx_config = {}
    ms_ctx_config = {}
    for k, v in kwargs.items():
        if k in MFContextConfig.get_supported_kwargs():
            mf_ctx_config[k] = v
        else:
            ms_ctx_config[k] = v
    ctx.mf_ctx_opr.set_context(**mf_ctx_config)
    ctx.ms_ctx_opr.set_context(**ms_ctx_config)


def get_context(attr_key):
    """
    Get context attribute value according to the input key.
    If some attributes are not set, they will be automatically obtained.

    Args:
        attr_key (str): The key of the attribute.

    Returns:
        Object, The value of given attribute key.

    Raises:
        ValueError: If input key is not an attribute in context.

    Examples:
        >>> from mindformers import build_context, get_context
        >>> config = {'context': {'mode': 'GRAPH_MODE'}, 'parallel':{}}
        >>> build_context(config=config)
        >>> get_context('max_device_memory')
    """
    ctx = Context()
    if attr_key in MFContextConfig.get_supported_kwargs():
        return getattr(ctx.mf_ctx_opr, attr_key, None)
    return ctx.ms_ctx_opr.get_context(attr_key)
