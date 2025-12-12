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
"""mcore glm4 moe model ST of inference"""
import argparse
import os
from transformers import AutoTokenizer

from mindspore.nn.utils import no_init_parameters

from tests.st.test_multi_cards_cases.test_model.utils import compare_distance

from mindformers import AutoModel, build_context, MindFormerConfig
from mindformers.core.parallel_config import build_parallel_config
from mindformers.tools.logger import logger


def test_glm4_moe_predict_mcore(device_num: int = 1):
    """
    Feature: Mcore Glm4Moe predict task
    Description: Two-card tp parallel
    Expectation: Success or assert precision failed
    """
    max_decode_length = 32
    config_path = os.path.join(os.path.dirname(__file__), "glm4_moe_infer.yaml")
    config = MindFormerConfig(config_path)
    config.use_parallel = device_num > 1
    config.parallel_config.model_parallel = device_num
    config.pretrained_model_dir = "/home/workspace/mindspore_dataset/weight/GLM-4.5-Air-tiny"
    # Reduced layer network
    config.model.model_config.num_hidden_layers = 2
    build_context(config)
    build_parallel_config(config)
    # Auto tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_dir)
    # init network
    with no_init_parameters():
        network = AutoModel.from_config(config)
        network.load_weights(config.pretrained_model_dir)
    # Build prompt and answer
    batch_datas = {1: {"prompt": "Please introduce some scenic spots in Beijing.",
                       "answer": "Please introduce some scenic spots in Beijing."
                                 "ahan flying UmbursalhsiqadereFINE写实ENVIRONMENTally NASpired "
                                 "Biosphericoux posit Lifts-offENS小的范围内"},
                   4: {"prompt": "Please introduce some scenic spots in Beijing.",
                       "answer": "Please introduce some scenic spots in Beijing."
                                 "ahan flying UmbursalhsiqadereFINE写实ENVIRONMENTally NASpired "
                                 "Biosphericoux posit Lifts-offENS小的范围内"},
                   }

    for batch_size, batch_data in batch_datas.items():
        input_ids = tokenizer.encode(batch_data["prompt"])
        input_ids_list = []
        answer = batch_data["answer"]
        for _ in range(0, batch_size):
            input_ids_list.append(input_ids)

        outputs = network.generate(input_ids_list,
                                   max_length=max_decode_length,
                                   do_sample=False,
                                   return_dict_in_generate=False)

        for output in outputs:
            output_text = tokenizer.decode(output)
            logger.info("test_glm4_5_air_predict, output_text: %s", str(output_text))
            compare_distance(output_text, answer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Glm4Moe ST")
    parser.add_argument("--device_num", type=int, default=2)

    args = parser.parse_args()
    os.environ['MS_ENABLE_LCCL'] = "off"
    os.environ['HCCL_DETERMINICTIC'] = "true"
    os.environ['LCCL_DETERMINICTIC'] = "1"
    os.environ['ASCEND_LAUNCH_BLOCKING'] = "1"
    os.environ['CUSTOM_MATMUL_SHUFFLE'] = "off"
    test_glm4_moe_predict_mcore(args.device_num)
