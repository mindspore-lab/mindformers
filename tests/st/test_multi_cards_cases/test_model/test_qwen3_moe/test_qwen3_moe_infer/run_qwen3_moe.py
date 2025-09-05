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
"""mcore qwen3-30b-a3b model ST of inference"""
import argparse
import os

from transformers import AutoTokenizer

from mindformers import AutoConfig
from mindformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
from mindformers import build_context, MindFormerConfig
from mindformers.tools.logger import logger

from tests.st.test_multi_cards_cases.test_model.utils import compare_distance


def test_qwen3_30b_a3b_predict_mcore(device_num: int = 1):
    """
    Feature: Mcore Qwen3-30B-A3B predict task
    Description: Two-card tp parallel
    Expectation: Success or assert precision failed
    """
    max_decode_length = 64
    config_path = os.path.join(os.path.dirname(__file__), "qwen3_moe_30b_a3b_infer.yaml")
    config = MindFormerConfig(config_path)
    config.use_parallel = device_num > 1
    build_context(config)
    # Auto tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_dir)
    # Auto config
    model_config = AutoConfig.from_pretrained(config_path)
    model_config.parallel_config.model_parallel = device_num
    # Reduced layer network
    model_config.num_hidden_layers = 2
    network = Qwen3MoeForCausalLM(model_config)
    # Load HF safetensors
    network.load_weights(config.pretrained_model_dir)
    # Build prompt and answer
    batch_datas = {1: {"prompt": "Give me a short introduction to large language model.",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                                 "Give me a short introduction to large language model.<|im_end|>\n"
                                 "<|im_start|>assistant\n使用網路.bootstrapcdn…the…the…the…the…the…the…the"
                                 "當您在../../../角落角落角落角落角落角落角落角落角落角落角落角落 Câm瀏瀏"
                                 "iska#__#__#__#__#__#__#__#__"},
                   4: {"prompt": "Please introduce some scenic spots in Beijing.",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                                 "Please introduce some scenic spots in Beijing.<|im_end|>\n<|im_start|>assistant\n"
                                 "使用網路電話及imageNamePREFIXesっきりtic不来/rem/rem/rem/rem/rem/rem/rem/rem/rem/rem "
                                 "… الفند.bootstrapcdn…the…the…the…the…the other reason reason reason reason(ver"
                                 "当地人(local.getLocal(local…)"},
                   }
    for batch_size, batch_data in batch_datas.items():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": batch_data["prompt"]}
        ]
        input_ids = tokenizer.encode(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ))
        input_ids_list = []
        answer = batch_data["answer"]
        for i in range(0, batch_size):
            input_ids_list.append(input_ids)

        outputs = network.generate(input_ids_list,
                                   max_length=max_decode_length,
                                   do_sample=False,
                                   return_dict_in_generate=False)

        for i in range(0, len(outputs)):
            output_text = tokenizer.decode(outputs[i])
            logger.info("test_qwen3_30b_a3b_predict, output_text:{}".format(str(output_text)))
            compare_distance(output_text, answer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Qwen3Moe ST")
    parser.add_argument("--device_num", type=int, default=2)

    args = parser.parse_args()
    test_qwen3_30b_a3b_predict_mcore(args.device_num)
