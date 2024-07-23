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
"""Llama2Network."""
from mindspore import log as logger
from mindformers import LlamaForCausalLM, LlamaTokenizer, MindFormerConfig, LlamaConfig, init_context, \
    TransformerOpParallelConfig


class Llama2Network:
    """Llama2Network."""
    @staticmethod
    def create_mfconfig(config_path, load_checkpoint=None, output_dir=None, world_size=None):
        """Create mindformers config for llama2 network for example."""
        config = MindFormerConfig(config_path)
        if load_checkpoint:
            config.load_checkpoint = load_checkpoint
        if output_dir:
            config.output_dir = output_dir
        if world_size:
            if config.parallel_config.model_parallel != world_size:
                logger.info(
                    f"world_size is {world_size}, not equal to \
                    config.parallel_config.model_parallel:{config.parallel_config.model_parallel}")
                config.parallel_config.model_parallel = world_size
                logger.info(f"reset config.parallel_config.model_parallel as :{world_size}")

        config.model.model_config.quant = ''
        config.model.model_config = LlamaConfig(**config.model.model_config)

        init_context(use_parallel=config.use_parallel, context_config=config.context, parallel_config=config.parallel)

        parallel_config = TransformerOpParallelConfig(**config.parallel_config)
        config.model.model_config.parallel_config = parallel_config
        return config

    @staticmethod
    def create_network(mindformers_config):
        network = LlamaForCausalLM(mindformers_config.model.model_config)
        network.set_train(False)
        network.phase = 'predict'
        return network

    @staticmethod
    def create_tokenizer(vocab_file):
        return LlamaTokenizer(vocab_file=vocab_file)
