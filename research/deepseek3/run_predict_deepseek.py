# Copyright 2025 Huawei Technologies Co., Ltd
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

"""DeepSeek-V3/R1 predict script"""

import argparse

import mindspore as ms
from mindspore import Model, Tensor
from mindspore.common import initializer

from mindformers import MindFormerConfig
from mindformers import build_context
from mindformers.tools.logger import logger
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.parallel_config import build_parallel_config
from mindformers.models.llama.llama_tokenizer_fast import LlamaTokenizerFast

from deepseek3_model_infer import InferenceDeepseekV3ForCausalLM
from deepseek3_config import DeepseekV3Config


def run_predict(args):
    """Deepseek-V3/R1 predict"""
    # inputs
    input_questions = [args.input]

    # set model config
    yaml_file = args.config
    config = MindFormerConfig(yaml_file)
    build_context(config)
    build_parallel_config(config)
    model_config = config.model.model_config
    model_config.parallel_config = config.parallel_config
    model_config.moe_config = config.moe_config
    model_config = DeepseekV3Config(**model_config)

    # build tokenizer
    tokenizer = LlamaTokenizerFast(config.processor.tokenizer.vocab_file,
                                   config.processor.tokenizer.tokenizer_file,
                                   unk_token=config.processor.tokenizer.unk_token,
                                   bos_token=config.processor.tokenizer.bos_token,
                                   eos_token=config.processor.tokenizer.eos_token,
                                   fast_tokenizer=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # build model from config
    network = InferenceDeepseekV3ForCausalLM(model_config)
    ms_model = Model(network)
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        seq_length = model_config.seq_length
        input_ids = Tensor(shape=(model_config.batch_size, seq_length), dtype=ms.int32, init=initializer.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, ms_model, network, infer_data, do_predict=True)

    inputs = tokenizer(input_questions, max_length=64, padding="max_length")["input_ids"]
    outputs = network.generate(inputs,
                               max_length=1024,
                               do_sample=False,
                               top_k=5,
                               top_p=1,
                               max_new_tokens=128)
    answer = tokenizer.decode(outputs)
    print("answer: ", answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='YAML config files, such as'
        './research/deepseek3/deepseek3_671b/predict_deepseek3_671b.yaml')
    parser.add_argument(
        '--input',
        type=str,
        default="生抽和老抽的区别是什么？")
    args_ = parser.parse_args()

    run_predict(args_)
