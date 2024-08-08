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
"""llama2 predict example."""
import argparse
import os

from mindspore import set_context
from mindspore.communication import init

from mindformers import MindFormerConfig, logger
from mindformers.core.context.build_context import set_predict_context_config
from mindformers.experimental.distri_cores.create_comm import initialize_model_parallel
from mindformers.experimental.infer.models.llama import ParallelLlamaForCausalLM
from mindformers.models.llama import LlamaConfig, LlamaTokenizer
from mindformers.tools import MODE
from mindformers.trainer.utils import transform_and_load_checkpoint


def main(config_path, use_parallel, load_checkpoint):
    # multi batch inputs
    inputs = ["I love Beijing, because",
              "LLaMA is a",
              "Huawei is a company that"]
    batch_size = len(inputs)

    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = use_parallel
    device_num = os.getenv('MS_WORKER_NUM')
    logger.info(f"Use device number: {device_num}, it will override config.model_parallel.")
    config.parallel_config.model_parallel = int(device_num) if device_num else 1
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.load_checkpoint = load_checkpoint

    # init context
    set_predict_context_config(config)
    mode = MODE.get(config.context.mode)
    set_context(mode=mode, max_device_memory=config.context.max_device_memory)

    # init communication
    init()
    initialize_model_parallel(tp_size=config.parallel_config.model_parallel,
                              order='tp')

    config.model.model_config.parallel_config = config.parallel_config
    config.model.model_config.batch_size = batch_size
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None
    model_name = config.trainer.model_name

    # build tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    # build model
    network = ParallelLlamaForCausalLM(model_config)

    # load checkpoint
    config.auto_trans_ckpt = False
    transform_and_load_checkpoint(config, network)

    inputs_ids = tokenizer(inputs, max_length=model_config.seq_length, padding="max_length")["input_ids"]
    outputs = network.generate(inputs_ids,
                               max_length=model_config.max_decode_length,
                               do_sample=model_config.do_sample,
                               top_k=model_config.top_k,
                               top_p=model_config.top_p)
    for output in outputs:
        print(tokenizer.decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='predict_llama2_7b.yaml', type=str,
                        help='model config file path.')
    parser.add_argument('--use_parallel', action='store_true',
                        help='if run model prediction in parallel mode.')
    parser.add_argument('--load_checkpoint', type=str,
                        help='load model checkpoint path or directory.')

    args = parser.parse_args()
    main(
        args.config_path,
        args.use_parallel,
        args.load_checkpoint
    )

# 多batch输出
# <s>I love Beijing,because it is a city that is constantly changing. I have been living here for 10 years ...
# <s>LlaMa is a large-scale, open-source, multimodal, multilingual, multitask, and multimodal pretrained ...
# <s>Huawei is a company that has been around for a long time. ...
