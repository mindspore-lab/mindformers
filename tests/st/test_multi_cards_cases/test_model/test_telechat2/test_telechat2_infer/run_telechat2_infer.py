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
"""mcore telechat2 model ST of inference"""
import argparse
import os
from transformers import AutoTokenizer

from mindspore.nn.utils import no_init_parameters

from mindformers import AutoModel, build_context, MindFormerConfig
from mindformers.core.parallel_config import build_parallel_config
from mindformers.tools.logger import logger


def test_telechat2_predict_mcore(device_num: int = 1):
    """
    Feature: Mcore TeleChat2 predict task
    Description: Two-card tp parallel
    Expectation: Success or assert precision failed
    """
    max_decode_length = 32
    config_path = os.path.join(os.path.dirname(__file__), "telechat2_infer.yaml")
    config = MindFormerConfig(config_path)
    config.use_parallel = device_num > 1
    config.parallel_config.model_parallel = device_num
    config.pretrained_model_dir = "/home/workspace/mindspore_dataset/weight/telechat2_7b"
    # Reduced layer network
    config.model.model_config.num_hidden_layers = 2
    build_context(config)
    build_parallel_config(config)
    # Auto tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_dir, trust_remote_code=True)
    # init network
    with no_init_parameters():
        network = AutoModel.from_config(config)
        network.load_weights(config.pretrained_model_dir)
    # Build prompt
    question = "Please introduce some scenic spots in Beijing."

    input_ids = tokenizer.encode(question)

    output = network.generate(input_ids,
                               max_length=max_decode_length,
                               do_sample=False,
                               return_dict_in_generate=False)

    output_text = tokenizer.decode(output[0])
    logger.info("test_telechat2_predict, output_text: %s", str(output_text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run TeleChat2 ST")
    parser.add_argument("--device_num", type=int, default=2)

    args = parser.parse_args()
    os.environ['MS_ENABLE_LCCL'] = "off"
    os.environ['HCCL_DETERMINICTIC'] = "true"
    os.environ['LCCL_DETERMINICTIC'] = "1"
    os.environ['ASCEND_LAUNCH_BLOCKING'] = "1"
    os.environ['CUSTOM_MATMUL_SHUFFLE'] = "off"
    test_telechat2_predict_mcore(args.device_num)
