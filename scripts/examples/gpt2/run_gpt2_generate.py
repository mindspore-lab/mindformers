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
"""gpt2 predict example."""
import argparse
import os

from mindformers.models.gpt2.gpt2_config import GPT2Config
from mindformers.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from mindformers.models.gpt2.gpt2 import GPT2LMHeadModel

from mindformers import MindFormerConfig, logger
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config

def main(config_path, use_parallel, load_checkpoint, load_tokenizer):
    # multi batch inputs
    inputs = ["I love Beijing, because",
              "GPT2 is a",
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

    os.environ["RUN_MODE"] = config.run_mode

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    config.model.model_config.batch_size = batch_size
    model_config = GPT2Config(**config.model.model_config)
    model_config.checkpoint_name_or_path = load_checkpoint

    # build tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(load_tokenizer)

    # build model
    network = GPT2LMHeadModel(model_config)

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
    parser.add_argument('--config_path', default='predict_gpt2_small_fp16.yaml', type=str,
                        help='model config file path.')
    parser.add_argument('--use_parallel', action='store_true',
                        help='if run model prediction in parallel mode.')
    parser.add_argument('--load_checkpoint', type=str,
                        help='load model checkpoint path or directory.')
    parser.add_argument('--load_tokenizer', type=str,
                        help='load tokenizer model directory.')

    args = parser.parse_args()
    main(
        args.config_path,
        args.use_parallel,
        args.load_checkpoint,
        args.load_tokenizer
    )