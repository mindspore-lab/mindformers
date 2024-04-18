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
import numpy as np

import mindspore as ms
from mindspore.communication.management import init

from mindformers import build_context, logger, build_parallel_config, GenerationConfig, AutoModel
from mindformers.models.build_config import build_model_config
from mindformers.tools.register.config import MindFormerConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.generation import logits_process

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
        npu_device_ids (list[int]):
            Get npu_device_ids from MindIE config.
    """

    def __init__(self, model_path, npu_mem_size, cpu_mem_size, block_size, rank_id=0, world_size=1,
                 npu_device_ids=None):
        self.config = None
        self.model_config = None
        self.generation_config = None

        # parallel predict with dynamic cluster.
        if world_size > 1:
            os.environ['MS_WORKER_NUM'] = str(world_size)
            os.environ['MS_ROLE'] = 'MS_WORKER'
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

        if world_size > 1:
            self.config.use_parallel = True

        if self.config and self.config.model.model_config:
            self.config.model.model_config.block_size = block_size
            self.model_config = build_model_config(self.config.model.model_config)

            self.num_layers = self.model_config.num_layers
            n_kv_heads = self.model_config.num_heads if self.model_config.n_kv_heads is None \
                else self.model_config.n_kv_heads
            head_dim = self.model_config.hidden_size // self.model_config.num_heads
            self.npu_num_blocks = world_size * (npu_mem_size * 1024 * 1024 * 1024) // \
                                  (block_size * n_kv_heads * head_dim * 2 * 2 * self.num_layers)
            self.cpu_num_blocks = world_size * (cpu_mem_size * 1024 * 1024 * 1024) // \
                                  (block_size * n_kv_heads * head_dim * 2 * 2 * self.num_layers)
            self.model_config.num_blocks = self.npu_num_blocks
            self.config.model.model_config.num_blocks = self.npu_num_blocks

            if self.config.use_parallel:
                build_parallel_config(self.config)
                self.config.model.model_config.checkpoint_name_or_path = None
                self.config.model.model_config.parallel_config = self.config.parallel_config

        self.generation_config = GenerationConfig.from_model_config(self.model_config)

        if not self.config.use_parallel and npu_device_ids:
            if len(npu_device_ids) != 1:
                raise ValueError("npu_device_ids should only contain one device_id")
            self.config.context.device_id = npu_device_ids[0]

        build_context(self.config)
        logger.info(f"Build context finished.")
        self.model = AutoModel.from_config(self.config)
        logger.info(f"Create model finished.")

        if self.config.use_parallel:
            ms_model = ms.Model(self.model)
            batch_size = self.model_config.batch_size
            seq_length = self.model_config.seq_length
            input_ids = np.ones(shape=tuple([batch_size, seq_length]))
            inputs = self.model.prepare_inputs_for_predict_layout(input_ids)
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
        gen_conf = kwargs.get("generation_config")
        if gen_conf is None:
            logits_processor = kwargs.get("logits_processor")
            process = self._get_logits_processor(logits_processor)
            kwargs.update({"generation_config": self.generation_config, "logits_processor": process})
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

    @staticmethod
    def _merge_processor_list(default_list, custom_list):
        """merge custom processor list with default list."""
        if not custom_list:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "logits_processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been padded to"
                        f" `.generate()`, but it has already been created with the values {default}."
                        f" {default} has been created by passing the corresponding arguments to generate or"
                        f" by the model's config default values. If you just want to change the default values"
                        f" of {object_type} consider passing them as arguments to `.generate()`"
                        f" instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list

    def _get_logits_processor(self, logits_processor):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        # initialize processors list
        processors = logits_process.LogitsProcessorList()

        if self.generation_config.repetition_penalty is not None and self.generation_config.repetition_penalty != 1.0:
            processors.append(logits_process.RepetitionPenaltyLogitsProcessor(
                repetition_penalty=self.generation_config.repetition_penalty))
        processors = self._merge_processor_list(processors, logits_processor)
        # `LogitNormalization` should always be the last logits processor, when present
        if self.generation_config.renormalize_logits is True:
            processors.append(logits_process.LogitNormalization())
        return processors
