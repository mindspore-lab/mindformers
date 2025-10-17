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
"""Parallel operator."""

from mindspore import context
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs

from mindformers.core.config_args import ParallelConfig
from mindformers.modules.transformer.transformer import (
    TransformerOpParallelConfig,
)
from mindformers.tools.logger import logger
from mindformers.tools.utils import set_strategy_save_path
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel


class ParallelOperator:
    """The wrapper of parallel operation."""

    def __init__(self, config):
        self.config = config
        parallel_ctx_config, parallel_config = self._handle_data()
        self.parallel_ctx = parallel_ctx_config
        self.parallel = ParallelConfig(**parallel_config)
        del self.config

    def _handle_data(self):
        if self.config.use_parallel:
            self._set_pipeline_stage()
        return self._get_parallel_ctx_config(), self._get_parallel_config()

    def _set_pipeline_stage(self):
        """Set pipeline stage number."""
        input_stages = self.config.parallel_config.pipeline_stage or 1
        if self.config.parallel.auto_pipeline:
            raise ValueError(
                "Automatic pipeline stage is unavailable for now, please set auto_pipeline to False in the yaml."
            )
        final_stages = input_stages

        self.config.parallel_config.pipeline_stage = final_stages
        if final_stages > 1:
            self.config.parallel.pipeline_stages = final_stages

    def _get_parallel_ctx_config(self):
        parallel_ctx = self.config.get('parallel', {})
        parallel_mode = parallel_ctx.get('parallel_mode')
        if parallel_mode not in [
                context.ParallelMode.SEMI_AUTO_PARALLEL,
                context.ParallelMode.AUTO_PARALLEL,
        ] and parallel_ctx.get('full_batch'):
            logger.info("full_batch is set to False for non-parallel modes")
            parallel_ctx['full_batch'] = False
        return parallel_ctx

    def _get_parallel_config(self):
        parallel_config = self.config.get('parallel_config', {})
        if isinstance(parallel_config, TransformerOpParallelConfig):
            parallel_config = parallel_config.to_dict()
        return ParallelConfig.filter_kwargs(**parallel_config)

    def init_communication(self):
        """Init communication services."""
        try:
            init()
        except Exception:
            logger.error(
                'Notice: if you are trying to run with a single device, '
                'please set use_parallel=False. '
                'If not, please check the error message above.'
            )
            raise
        device_num = get_group_size()
        self.parallel_ctx['device_num'] = device_num
        context.reset_auto_parallel_context()
        set_strategy_save_path(self.parallel_ctx)
        self._set_ms_auto_parallel_context(**self.parallel_ctx)
        self._set_ms_parallel()
        self._set_manmul_parallel()
        return get_rank(), device_num

    @staticmethod
    def _set_ms_auto_parallel_context(**parallel_ctx):
        full_batch = parallel_ctx.get('full_batch')
        src_ds_stra = parallel_ctx.pop('dataset_strategy', None)
        if not full_batch and isinstance(src_ds_stra, list):
            # convert the type of dataset_strategy from list to tuple
            ds_stra = tuple(tuple(ds_item) for ds_item in src_ds_stra)
            parallel_ctx['dataset_strategy'] = ds_stra
        context.set_auto_parallel_context(**parallel_ctx)

    @staticmethod
    def _set_ms_parallel():
        """Init parallel config of mindspore."""
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode == "auto_parallel":
            set_algo_parameters(
                elementwise_op_strategy_follow=False, fully_use_devices=False
            )
        else:
            set_algo_parameters(
                elementwise_op_strategy_follow=True, fully_use_devices=True
            )
        _set_multi_subgraphs()

    def _set_manmul_parallel(self):
        """ Init parallel for manmul parallel. """
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode != "stand_alone":
            return

        model_parallel = getattr(self.parallel, "model_parallel", 1)
        data_parallel = getattr(self.parallel, "data_parallel", 1)
        pipeline_stage = getattr(self.parallel, "pipeline_stage", 1)
        expert_parallel = getattr(self.parallel, "expert_parallel", 1)
        order = "tp-ep-pp-dp"

        parallel_strategy = ""
        parallel_map = {
            "TP": model_parallel,
            "DP": data_parallel,
            "EP": expert_parallel,
            "PP": pipeline_stage,
        }
        for parallel_type, parallel_size in parallel_map.items():
            if parallel_size > 1:
                parallel_strategy += f"{parallel_type}{parallel_size}"
        logger.info(f"Initializing model parallel with strategy {parallel_strategy} and order {order}.")

        initialize_model_parallel(
            tensor_model_parallel_size=model_parallel,
            data_parallel_size=data_parallel,
            pipeline_model_parallel_size=pipeline_stage,
            expert_model_parallel_size=expert_parallel,
            order=order
        )
