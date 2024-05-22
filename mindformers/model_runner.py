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
from typing import Optional, List, Union
import numpy as np

import mindspore as ms
from mindspore import context, ops, Tensor
from mindspore.communication.management import init
from mindspore.common.initializer import Zero
from mindspore._c_expression import swap_cache

from mindformers import models
from mindformers import build_context, logger, build_parallel_config, GenerationConfig, AutoModel, AutoConfig
from mindformers.models.build_config import build_model_config
from mindformers.models.utils import convert_mstype
from mindformers.tools.register.config import MindFormerConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.generation.logits_process import LogitsProcessorAcceleration

__all__ = ["ModelRunner"]

class ModelRunner:
    """
    ModelRunner API, supports MF to be a backend of MindIEServer.

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

    Returns:
        A MindIERunner object.
    """

    def __new__(cls, model_path, npu_mem_size, cpu_mem_size, block_size, rank_id=0, world_size=1,
                npu_device_ids=None):
        experiment_mode, config = _get_model_config(model_path)
        model_type = config.model.arch.type if not experiment_mode else config.architectures[0]
        logger.info(f"The model type is: {model_type}")
        model_runner_cls = MindIEModelRunner
        if model_type not in models.__all__:
            try:
                model_runner_cls = __import__(model_type, ["MindIEModelRunner"]).MindIEModelRunner
            except ImportError:
                logger.info(f"import MindIEModelRunner from module {model_type} failed, "
                            f"and will use the default one defined in mindformers.")

        model_runner = model_runner_cls(experiment_mode, config, npu_mem_size, cpu_mem_size, block_size, rank_id,
                                        world_size, npu_device_ids)
        return model_runner


class MindIEModelRunner:
    """
    Implementation of ModelRunner.

    Args:
        experiment_mode (bool):
            Is experiment model.
        model_config (PretrainedConfig):
            Model config.
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

    def __init__(self, experiment_mode, model_config, npu_mem_size, cpu_mem_size, block_size, rank_id=0, world_size=1,
                 npu_device_ids=None):
        self.experiment_mode = experiment_mode
        self.config = None if experiment_mode else model_config
        self.post_processor = LogitsProcessorAcceleration()

        # parallel predict with dynamic cluster.
        if world_size > 1:
            os.environ['MS_WORKER_NUM'] = str(world_size)
            os.environ['MS_ROLE'] = 'MS_WORKER'
            if rank_id == 0 and os.fork() == 0:
                os.environ['MS_ROLE'] = 'MS_SCHED'
                init()

        if world_size > 1:
            if not self.experiment_mode:
                self.config.use_parallel = True
            else:
                raise SystemError(f"You are running in experiment mode. World size can only be 1, "
                                  f"but got world_size = {world_size}")

        self.model_config = build_model_config(self.config.model.model_config) if not experiment_mode else model_config
        self.num_layers = self.model_config.num_layers
        n_kv_heads = self.model_config.num_heads if self.model_config.n_kv_heads is None \
            else self.model_config.n_kv_heads
        n_kv_heads = n_kv_heads // world_size  # check the divisibility in model initialization.
        head_dim = self.model_config.hidden_size // self.model_config.num_heads
        self.npu_num_blocks = (npu_mem_size * 1024 * 1024 * 1024) // \
                              (block_size * n_kv_heads * head_dim * 2 * 2 * self.num_layers)
        self.cpu_num_blocks = (cpu_mem_size * 1024 * 1024 * 1024) // \
                              (block_size * n_kv_heads * head_dim * 2 * 2 * self.num_layers)
        self.model_config.num_blocks = self.npu_num_blocks
        self.model_config.block_size = block_size

        self.generation_config = GenerationConfig.from_model_config(self.model_config)

        if not self.experiment_mode:
            if self.config.use_parallel:
                build_parallel_config(self.config)
                self.config.model.model_config.checkpoint_name_or_path = None
                self.config.model.model_config.parallel_config = self.config.parallel_config

            if not self.config.use_parallel and npu_device_ids:
                if len(npu_device_ids) != 1:
                    raise ValueError("npu_device_ids should only contain one device_id")
                self.config.context.device_id = npu_device_ids[0]

            build_context(self.config)
            logger.info(f"Build context finished.")
            self.config.model.model_config.block_size = block_size
            self.config.model.model_config.num_blocks = self.npu_num_blocks
            self.model = AutoModel.from_config(self.config)
            logger.info(f"Create model finished.")

            if self.config.use_parallel:
                ms_model = ms.Model(self.model)
                batch_size = self.model_config.batch_size
                seq_length = self.model_config.seq_length
                input_ids = np.ones(shape=tuple([batch_size, seq_length]))
                inputs = self.model.prepare_inputs_for_predict_layout(input_ids)
                transform_and_load_checkpoint(self.config, ms_model, self.model, inputs, do_predict=True)
        else:
            context.set_context(mode=0, device_id=npu_device_ids[0])
            logger.info(f"Build context finished.")
            self.model = AutoModel.from_config(self.model_config, trust_remote_code=True)
            logger.info(f"Create model finished.")

        if self.model_config.is_dynamic:
            self.model.set_dynamic_inputs()

        cpu_kv_shape = (self.cpu_num_blocks, block_size, n_kv_heads, head_dim)
        compute_dtype = convert_mstype(self.model_config.compute_dtype)
        self.key_host = [ms.Parameter(ms.Tensor(shape=cpu_kv_shape, dtype=compute_dtype, init=Zero()),
                                      name=f"key_host_{i}", requires_grad=False) for i in range(self.num_layers)]
        self.value_host = [ms.Parameter(ms.Tensor(shape=cpu_kv_shape, dtype=compute_dtype, init=Zero()),
                                        name=f"value_host_{i}", requires_grad=False) for i in range(self.num_layers)]

    def forward(self, input_ids: [Union[List[int], List[List[int]]]],
                valid_length_each_example: List[int],
                is_finished: List[bool],
                block_tables: Optional[Tensor] = None,
                slot_mapping: Optional[Tensor] = None,
                prefill: bool = True,
                generation_config: [GenerationConfig] = None):
        """
        Call self.model.infer() or self.model.forward() to do infer and return logits on next position, \
        can choose do prefill or decode predict.

        Args:
            input_ids (List(List(int))):
                Input ids after padding.
            valid_length_each_example (List(int)):
                Valid input length except padding.
            is_finished (List(bool)):
                Whether each sequence is finished its generation.
            block_tables (Tensor):
                Params for page attention
            slot_mapping (Tensor):
                Params for page attention
            prefill (bool):
                Whether to do prefill predict or decode predict
            generation_config (`GenerationConfig`):
                The generation configuration to be used as base parametrization for the generation call.

        Returns:
            next_token, is_finished
        """
        seed = 0 if (self.config is None or self.config.seed is None) else self.config.seed
        seed = seed if (generation_config is None or generation_config.get("seed") is None) else \
            generation_config.get("seed")[0]
        np.random.seed(seed)

        input_ids_seq_length = max(valid_length_each_example)
        logits_processor = self.model.get_logits_processor(self.generation_config, input_ids_seq_length, None)
        logits_warper = self.model.get_logits_warper(self.generation_config)

        kwargs = {"input_ids": input_ids,
                  "valid_length_each_example": valid_length_each_example,
                  "generation_config": self.generation_config,
                  "logits_processor": logits_processor,
                  "logits_warper": logits_warper,
                  "block_tables": block_tables,
                  "slot_mapping": slot_mapping,
                  "prefill": prefill,
                  "is_finished": is_finished}
        if generation_config is None:
            return self.model.infer(**kwargs)

        do_sample = generation_config.get("do_sample")
        temperature = generation_config.get("temperature")
        top_k = generation_config.get("top_k")
        top_p = generation_config.get("top_p")
        repetition_penalty = generation_config.get("repetition_penalty")

        # Check if the sample parameters are all the same, \
        # because the post_process in text_generator does not support different ones.
        if _the_same_sample_parmas(generation_config):
            generation_config = {"do_sample": do_sample[0], "temperature": temperature[0], "top_k": int(top_k[0]),
                                 "top_p": top_p[0], "repetition_penalty": repetition_penalty[0]}
            self.generation_config.update(**generation_config)
            logits_processor = self.model.get_logits_processor(self.generation_config, input_ids_seq_length, None)
            logits_warper = self.model.get_logits_warper(self.generation_config)
            kwargs["generation_config"] = self.generation_config
            kwargs["logits_processor"] = logits_processor
            kwargs["logits_warper"] = logits_warper
            return self.model.infer(**kwargs)

        res, current_idx = self.model.forward(**kwargs)
        logits = ops.reshape(res[0], (-1, res[0].shape[-1]))
        if prefill and logits.shape[0] > len(current_idx):
            logits = logits[Tensor(current_idx)]

        ori_logits = logits
        if np.any(temperature != 1.0):
            logits = self.post_processor.temperature_process(logits, temperature)
        if np.any(top_k != 1):
            logits = self.post_processor.top_k_process(logits, top_k)
        if np.any(top_p != 1.0):
            logits = self.post_processor.top_p_process(logits, top_p)
        if self.generation_config.renormalize_logits:
            logits = self.post_processor.logits_normalization(logits)

        p_norms = ops.softmax(logits).asnumpy()
        batch_size = np.array(input_ids).shape[0]
        length = logits.shape[1]

        next_ids = [None] * batch_size
        for i in range(batch_size):
            if is_finished[i]:
                continue
            if do_sample[i]:
                p_norm = p_norms[i]
                target = np.random.choice(length, p=p_norm)
            else:
                p_norm = ori_logits[i]
                target = ops.argmax(p_norm)
            next_ids[i] = int(target)

        return next_ids, is_finished

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

    def swap(self, block_tables, swap_type):
        """
        Swap key/value cache between host and device, to support multi-batch and long-sequence inference.

        Args:
            block_tables:
                A 2-D array contains src and dst blocks to swap.
            swap_type:
                A bool value indicating the data direction: "True" for device-to-host, and "False" for host-to-device.
        """
        for i in range(self.num_layers):
            key_cache, value_cache = self.model.kvcache(i)
            swap_cache(self.key_host[i], key_cache, ms.Tensor(block_tables), swap_type)
            swap_cache(self.value_host[i], value_cache, ms.Tensor(block_tables), swap_type)


def _the_same_sample_parmas(generation_config):
    """
    Check if the sample parameters are all the same.

    Args:
        generation_config: dict of generation config contains dosample parameters.

    Returns:
         Bool value indicates if all sample parameters the same.
    """

    do_sample = generation_config.get("do_sample")
    temperature = generation_config.get("temperature")
    top_k = generation_config.get("top_k")
    top_p = generation_config.get("top_p")
    repetition_penalty = generation_config.get("repetition_penalty")
    return np.all(do_sample == do_sample[0]) and np.all(temperature == temperature[0]) and np.all(top_k == top_k[0]) \
            and np.all(top_p == top_p[0]) and np.all(repetition_penalty == repetition_penalty[0])


def _get_model_config(model_path):
    """
    Get model config from the config file.

    Args:
        model_path: path of model config file.

    Returns:
         experiment_mode, model_config.
    """

    experiment_mode = False
    if os.path.isdir(model_path):
        json_list = [file for file in os.listdir(model_path)
                     if file.endswith("config.json")]
        yaml_list = [file for file in os.listdir(model_path)
                     if file.endswith(".yaml")]
        if yaml_list:
            yaml_path = os.path.join(model_path, yaml_list[0])
            model_config = MindFormerConfig(yaml_path)
        elif json_list:
            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            experiment_mode = True
        else:
            raise FileNotFoundError(f"There is no yaml file nor config.json file for model config in {model_path}.")
    else:
        raise ValueError(f"The path {model_path} is not exist.")

    return experiment_mode, model_config
