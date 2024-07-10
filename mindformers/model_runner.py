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
from typing import Optional, List, Union, Dict
import numpy as np

import mindspore as ms
from mindspore import ops, Tensor
from mindspore.communication.management import init
from mindspore.common.initializer import Zero
from mindspore._c_expression import swap_cache

from mindformers import models, MindFormerRegister, MindFormerModuleType
from mindformers import build_context, logger, build_parallel_config, GenerationConfig
from mindformers import AutoModel, AutoConfig, AutoTokenizer
from mindformers.models.utils import convert_mstype, ms_type_to_str
from mindformers.tools.register.config import MindFormerConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.tools.hub.dynamic_module_utils import get_class_from_dynamic_module

__all__ = ["get_model", "ModelRunner"]


def register_auto_class(config, pretrained_model_name_or_path, class_type, use_fast=True):
    """convert to auto class"""
    if config.model.model_config.auto_map:
        class_auto = config["model"]["model_config"]["auto_map"]
        if class_type == "AutoConfig" and \
            config.model.model_config.type not in MindFormerRegister.registry[MindFormerModuleType.CONFIG]:
            class_ref = class_auto[class_type]
            config_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path)
            MindFormerRegister.register_cls(config_class, module_type=MindFormerModuleType.CONFIG)

        if class_type == "AutoTokenizer" and \
            config.processor.tokenizer.type not in MindFormerRegister.registry[MindFormerModuleType.TOKENIZER]:
            if use_fast and class_auto[class_type][1] is not None:
                class_ref = class_auto[class_type][1]
            else:
                class_ref = class_auto[class_type][0]
            tokenizer_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path)
            MindFormerRegister.register_cls(tokenizer_class, module_type=MindFormerModuleType.TOKENIZER)

        if class_type == "AutoModel" and \
            config.model.arch.type not in MindFormerRegister.registry[MindFormerModuleType.MODELS]:
            class_ref = class_auto[class_type]
            model_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path)
            MindFormerRegister.register_cls(model_class, module_type=MindFormerModuleType.MODELS)


def get_model(model_name_or_path: str,
              revision: Optional[str] = None,
              trust_remote_code: Optional[bool] = True,
              **kwargs):
    """
    get_model API, supports MF to be a backend of MindIEServer.

    Args:
        model_name_or_path (str):
            A path to a *directory* containing vocabulary files() required by the tokenizer.
        revision (`str`, *optional*, defaults to `"None"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id.
        trust_remote_code (`bool`, *optional*, defaults to `True`):
            Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional key word arguments for AutoTokenizer.from_pretrained.

    Returns:
        A Tokenizer object and others.
    """
    if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
        logger.debug(f"model_name_or_path is {model_name_or_path}")
        config_path = _get_model_config(model_name_or_path)
        config = MindFormerConfig(config_path)
        model_type = config.model.arch.type
        logger.info(f"The model type is: {model_type}")
        register_auto_class(config, model_name_or_path, class_type="AutoTokenizer")

        use_fast = kwargs.get("use_fast", True)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision,
                                                  trust_remote_code=trust_remote_code,
                                                  use_fast=use_fast)
    else:
        raise ValueError(f"{model_name_or_path} does not exist or is not a directory.")

    input_builder = InputBuilder(tokenizer)
    return tokenizer, input_builder


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
        config_path = _get_model_config(model_path)
        config = MindFormerConfig(config_path)
        model_type = config.model.arch.type
        logger.info(f"The model type is: {model_type}")
        model_runner_cls = MindIEModelRunner
        if model_type not in models.__all__:
            try:
                model_runner_cls = __import__(model_type, ["MindIEModelRunner"]).MindIEModelRunner
            except ImportError:
                logger.info(f"import MindIEModelRunner from module {model_type} failed, "
                            f"and will use the default one defined in mindformers.")

        model_runner = model_runner_cls(model_path, config_path, npu_mem_size, cpu_mem_size,
                                        block_size, rank_id, world_size, npu_device_ids)
        return model_runner


class MindIEModelRunner:
    """
    Implementation of ModelRunner.

    Args:
        model_path(str):
            The model config path contains model config file and tokenizer file.
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

    def __init__(self, model_path, config_path, npu_mem_size, cpu_mem_size, block_size, rank_id=0,
                 world_size=1, npu_device_ids=None):
        self.config = MindFormerConfig(config_path)
        # register to Auto Class
        register_auto_class(self.config, model_path, class_type="AutoConfig")
        register_auto_class(self.config, model_path, class_type="AutoTokenizer")
        register_auto_class(self.config, model_path, class_type="AutoModel")

        # parallel predict with dynamic cluster.
        if world_size > 1:
            self.config.use_parallel = True
            os.environ['MS_WORKER_NUM'] = str(world_size)
            os.environ['MS_ROLE'] = 'MS_WORKER'
            if rank_id == 0 and os.fork() == 0:
                os.environ['MS_ROLE'] = 'MS_SCHED'
                init()

        self.model_config = AutoConfig.from_pretrained(config_path)
        self.num_layers = self.model_config.num_layers
        self.num_kv_heads = self.model_config.num_heads if self.model_config.n_kv_heads is None \
            else self.model_config.n_kv_heads
        self.num_kv_heads = self.num_kv_heads // world_size  # check the divisibility in model initialization.
        self.head_size = self.model_config.hidden_size // self.model_config.num_heads
        self.npu_num_blocks = (npu_mem_size * 1024 * 1024 * 1024) // \
                              (block_size * self.num_kv_heads * self.head_size * 2 * 2 * self.num_layers)
        self.cpu_num_blocks = (cpu_mem_size * 1024 * 1024 * 1024) // \
                              (block_size * self.num_kv_heads * self.head_size * 2 * 2 * self.num_layers)
        self.model_config.block_size = block_size
        self.model_config.num_blocks = self.npu_num_blocks
        self.model_config.checkpoint_name_or_path = None
        if not hasattr(self.model_config, "max_position_embedding") or not self.model_config.max_position_embedding:
            self.model_config.max_position_embedding = self.model_config.seq_length

        self.generation_config = GenerationConfig.from_model_config(self.model_config)

        if self.config.use_parallel:
            build_parallel_config(self.config)
            self.model_config.parallel_config = self.config.parallel_config

        if not self.config.use_parallel and npu_device_ids:
            if len(npu_device_ids) != 1:
                raise ValueError("npu_device_ids should only contain one device_id")
            self.config.context.device_id = npu_device_ids[0]

        build_context(self.config)
        logger.info(f"Build context finished.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        logger.info(f"Build tokenizer finished.")
        self.model = AutoModel.from_config(self.model_config)
        logger.info(f"Build model finished.")

        ms_model = ms.Model(self.model)
        batch_size = self.model_config.batch_size
        seq_length = self.model_config.seq_length
        input_ids = np.ones(shape=tuple([batch_size, seq_length]))
        inputs = self.model.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(self.config, ms_model, self.model, inputs, do_predict=True)
        logger.info(f"Load checkpoints finished.")

        if self.model_config.is_dynamic:
            self.model.set_dynamic_inputs()

        compute_dtype = convert_mstype(self.model_config.compute_dtype)
        self.dtype = ms_type_to_str[compute_dtype]
        cpu_kv_shape = (self.cpu_num_blocks, block_size, self.num_kv_heads, self.head_size)
        self.key_host = [ms.Parameter(ms.Tensor(shape=cpu_kv_shape, dtype=compute_dtype, init=Zero()),
                                      name=f"key_host_{i}", requires_grad=False) for i in range(self.num_layers)]
        self.value_host = [ms.Parameter(ms.Tensor(shape=cpu_kv_shape, dtype=compute_dtype, init=Zero()),
                                        name=f"value_host_{i}", requires_grad=False) for i in range(self.num_layers)]

    def forward(self, input_ids: [Union[List[int], List[List[int]]]],
                valid_length_each_example: List[int],
                block_tables: Optional[Tensor] = None,
                slot_mapping: Optional[Tensor] = None,
                prefill: bool = True,):
        """
        Call self.model.infer() or self.model.forward() to do infer and return logits on next position, \
        can choose do prefill or decode predict.

        Args:
            input_ids (List(List(int))):
                Input ids after padding.
            valid_length_each_example (List(int)):
                Valid input length except padding.
            block_tables (Tensor):
                Params for page attention
            slot_mapping (Tensor):
                Params for page attention
            prefill (bool):
                Whether to do prefill predict or decode predict

        Returns:
            logits (Tensor)
        """

        res, current_idx = self.model.forward(input_ids=input_ids,
                                              valid_length_each_example=valid_length_each_example,
                                              block_tables=block_tables,
                                              slot_mapping=slot_mapping,
                                              prefill=prefill,
                                              use_past=True)
        if isinstance(res, tuple):
            logits = ops.reshape(res[0], (-1, res[0].shape[-1]))
        else:
            logits = ops.reshape(res, (-1, res.shape[-1]))
        if prefill and logits.shape[0] > len(current_idx):
            logits = logits[Tensor(current_idx)]

        return logits

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


def _get_model_config(model_path):
    """
    Get model config from the config file.

    Args:
        model_path: path of model config file.

    Returns:
        config_path.
    """

    if os.path.isdir(model_path):
        yaml_list = [file for file in os.listdir(model_path)
                     if file.endswith(".yaml")]
        if yaml_list:
            yaml_path = os.path.join(model_path, yaml_list[0])
        else:
            raise FileNotFoundError(f"There is no yaml file for model config in {model_path}.")
    else:
        raise ValueError(f"The path {model_path} is not exist.")

    return yaml_path


class InputBuilder:
    """
    Implementation of InputBuilder.

    Args:
        tokenizer (PreTrainedTokenizer):
            A tokenizer for text processing.
        chat_template (str):
            A Jinja template to use for this conversion.
        system_role_name (str):
            The name of system role.
        user_role_name (str):
            The name of user role.
        max_length (int):
            The max length of input tokens.
    """

    def __init__(self, tokenizer, chat_template="", system_role_name="system", user_role_name="user", max_length=2048):
        self.tokenizer = tokenizer
        self.system_role_name = system_role_name
        self.user_role_name = user_role_name
        if chat_template:
            self.tokenizer.chat_template = chat_template
        self.max_length = max_length
        self.rank = 0
        self.adapt_to_max_length = False

    def make_context(self, rank: int, conversation: List[Dict[str, str]], add_generation_prompt: bool = True,
                     adapt_to_max_length: bool = False, **kwargs):
        """
        Make a conversation tokens. Adapt interface of mindie-llm.

        Args:
            rank (int):
                The rank id.
            conversation (List[Dict[str, str]]):
                A conversation object or list of dicts.
            add_generation_prompt (bool, *optional*):
                Whether to end the prompt with the token(s) that indicate the start of an assistant message.
            adapt_to_max_length (bool, *optional*):
                Where input tokens should less max_length.

        Returns:
             context_tokens
        """
        self.rank = rank
        self.adapt_to_max_length = adapt_to_max_length
        context_tokens = self._apply_chat_template(conversation, add_generation_prompt=add_generation_prompt,
                                                   **kwargs)
        return context_tokens

    def _apply_chat_template(self, conversation: List[Dict[str, str]], **kwargs):
        """
        Converts a Conversation to a list of token ids.

        Args:
            conversation (List[Dict[str, str]]):
                A conversation object or list of dicts.

        Returns:
             input_ids
        """
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("The tokenizer dose not implement apply_chat_template function.")
        if not self.tokenizer.chat_template:
            raise RuntimeError("The model does not appear to be a chat model because it is not configured with a "
                               "`chat_template`.")
        input_ids = self.tokenizer.apply_chat_template(conversation, **kwargs)
        return input_ids
