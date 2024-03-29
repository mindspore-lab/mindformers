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

"""
For text generation
"""
import os

import mindspore as ms
from mindspore.communication.management import init

from mindformers import build_context, logger, build_parallel_config, GenerationConfig, AutoModel
from mindformers.models.build_config import build_model_config
from mindformers.tools.register.config import MindFormerConfig
from mindformers.trainer.utils import transform_and_load_checkpoint

__all__ = ["ModelRunner"]


class ModelRunner:
    """
    ModelRunner supports MF to be a backend of MindIEServer.

    Args:
        model_path (str):
            Model path contains a yaml file for model configuration.
        npu_mem_size (int):
            Npu memory size used for kv-cache.
        cpu_mem_size (int):
            Cpu memory size used for kv-cache.
        block_size (int):
            Block size used for kv-cache.
        rank_id (int):
            Rank id used for infer.
        world_size (int):
            Rank size used for infer.
        ip (str):
            The host ip address, only needed for parallel-predict.
        port (str):
            The port num ranged from 1024 to 65536, only needed for parallel-predict.
    """

    def __init__(self, model_path, npu_mem_size, cpu_mem_size, block_size, rank_id=0, world_size=1, ip=None, port=None):
        self.config = None
        self.model_config = None
        self.generation_config = None

        # parallel predict with dynamic cluster.
        if world_size > 1:
            os.environ['MS_WORKER_NUM'] = str(world_size)
            os.environ['MS_ROLE'] = 'MS_WORKER'
            os.environ['MS_SCHED_HOST'] = ip
            os.environ['MS_SCHED_PORT'] = port
            if rank_id == 0 and os.fork() == 0:
                os.environ['MS_ROLE'] = 'MS_SCHED'
                init()

        if os.path.isdir(model_path):
            yaml_list = [file for file in os.listdir(model_path)
                         if file.endswith(".yaml")]
            if not yaml_list:
                raise FileNotFoundError(f"There is no yaml file for model config in {model_path}.")
            yaml_path = os.path.join(model_path, yaml_list[0])
            self.config = MindFormerConfig(yaml_path)
        else:
            raise ValueError(f"The path {model_path} is not exist.")

        if self.config and self.config.model.model_config:
            self.config.model.model_config.block_size = block_size
            self.model_config = build_model_config(self.config.model.model_config)

            self.num_layers = self.model_config.num_layers
            n_kv_heads = self.model_config.num_heads if self.model_config.n_kv_heads is None \
                else self.model_config.n_kv_heads
            head_dim = self.model_config.hidden_size // self.model_config.num_heads
            self.npu_num_blocks = (npu_mem_size * 1024 * 1024 * 1024) // \
                                  (block_size * n_kv_heads * head_dim * 2 * 2 * self.num_layers)
            self.cpu_num_blocks = (cpu_mem_size * 1024 * 1024 * 1024) // \
                                  (block_size * n_kv_heads * head_dim * 2 * 2 * self.num_layers)
            self.model_config.num_blocks = self.npu_num_blocks
            self.config.model.model_config.num_blocks = self.npu_num_blocks

            if self.config.use_parallel:
                build_parallel_config(self.config)
                self.config.model.model_config.checkpoint_name_or_path = None
                self.config.model.model_config.parallel_config = self.config.parallel_config

        self.generation_config = GenerationConfig.from_model_config(self.model_config)

        build_context(self.config)
        logger.info(f"Build context finished.")
        self.model = AutoModel.from_config(self.config)
        logger.info(f"Create model finished.")

        if self.config.use_parallel:
            ms_model = ms.Model(self.model)
            inputs = self.model.prepare_inputs_for_export(True)
            transform_and_load_checkpoint(self.config, ms_model, self.model, inputs, do_predict=True)

        if self.model_config.is_dynamic:
            self.model.set_dynamic_inputs()

    def forward(self, **kwargs):
        """
        Call self.model.infer() to do infer and return logits on next position, can choose do prefill or decode predict.

        Args:
            **kwargs:
                Refers to GenerationMixin.infer().

        Returns:
            next_token, is_finished
        """
        return self.model.infer(**kwargs)

    def generate(self, **kwargs):
        """
        Call self.model.generate() to generate the words according to the given the input ids.

        Args:
            **kwargs:
                Refers to GenerationMixin.generate().

        Returns:
            A list of the generated token ids.
        """
        return self.model.generate(**kwargs)
