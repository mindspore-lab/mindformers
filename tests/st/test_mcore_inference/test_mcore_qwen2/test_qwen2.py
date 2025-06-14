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
"""mcore qwen2.5-3b model ST of inference"""
import math
import os

import pytest
import jieba
import numpy as np

from mindformers import build_context, MindFormerConfig
from mindformers.models.qwen2.configuration_qwen2 import Qwen2Config
from mindformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer


def _get_all_words(standard_cut_infer_ret_list, test_cut_infer_ret_list):
    all_words = []
    for s_cut in standard_cut_infer_ret_list:
        if s_cut not in all_words:
            all_words.append(s_cut)
    for t_cut in test_cut_infer_ret_list:
        if t_cut not in all_words:
            all_words.append(t_cut)
    return all_words


def _get_word_vector(standard_cut_infer_ret_list, test_cut_infer_ret_list, all_words):
    la_standard = []
    lb_test = []
    for word in all_words:
        la_standard.append(standard_cut_infer_ret_list.count(word))
        lb_test.append(test_cut_infer_ret_list.count(word))
    return la_standard, lb_test


def _get_calculate_cos(la_standard, lb_test):
    laa = np.array(la_standard)
    lbb = np.array(lb_test)
    cos = (np.dot(laa, lbb.T)) / ((math.sqrt(np.dot(laa, laa.T))) * (math.sqrt(np.dot(lbb, lbb.T))))
    return np.round(cos, 2)


def generate_input_ids(batch_size, input_seq_length):
    return [[i + 1 for i in range(input_seq_length)]
            for _ in range(batch_size)]


def compare_distance(x1, x2, bench_sim=0.95):
    """compare distance"""
    y1 = list(jieba.cut(x1))
    y2 = list(jieba.cut(x2))
    all_words = _get_all_words(y1, y2)
    laa, lbb = _get_word_vector(y1, y2, all_words)
    sim = _get_calculate_cos(laa, lbb)
    print("calculate sim is:{}".format(str(sim)))
    assert sim >= bench_sim


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_qwen2_5_3b_predict_mcore():
    """
    Feature: Infer interface
    Description: Test mcore interface for prediction.
    Expectation: AssertionError
    """
    config_path = "predict_qwen2_5_3b_instruct.yaml"
    load_safetensors = "/home/workspace/mindspore_dataset/weight/Qwen2.5-3B-Instruct"

    max_decode_length = 128

    config = MindFormerConfig(config_path)
    config.parallel_config.model_parallel = 1
    config.use_parallel = False
    config.load_checkpoint = load_safetensors
    config.context.device_id = int(os.environ.get("DEVICE_ID", "0"))

    build_context(config)
    model_config = Qwen2Config(**config.model.model_config)
    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(load_safetensors)
    # build model
    network = Qwen2ForCausalLM(model_config)

    if config.load_checkpoint:
        network.load_weights(load_safetensors)
    batch_datas = {1: {"prompt": "你好！",
                       "answer": "<|im_start|>system\n" +
                                 "You are a helpful assistant.<|im_end|>\n" +
                                 "<|im_start|>user\n" +
                                 "你好！<|im_end|>\n" +
                                 "<|im_start|>assistant\n" +
                                 "你好！<|im_end|>"},
                   4: {"prompt": "用python编写快速排序",
                       "answer": "<|im_start|>system\n" +
                                 "You are a helpful assistant.<|im_end|>\n" +
                                 "<|im_start|>user\n" +
                                 "用python编写快速排序<|im_end|>\n" +
                                 "<|im_start|>assistant\n" +
                                 "快速排序是一种高效的排序算法，采用分治策略来把一个序列分为较小和较大的两个子序列，"
                                 "然后递归地排序两个子序列。下面是一个使用Python实现的快速排序算法：\n" +
                                 "\n" +
                                 "```python\n" +
                                 "def quick_sort(arr):\n" +
                                 "    if len(arr) <= 1:\n" +
                                 "        return arr\n" +
                                 "    else:\n" +
                                 "        pivot = arr[len(arr) // 2]\n" +
                                 "        left = [x for x in arr if x < pivot]\n" +
                                 "        middle = [x for x in arr if x == pivot\n"},
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
            print("test_qwen2_5_3b_predict_standalone, output_text:", output_text)
            compare_distance(output_text, answer)

    batch_size = 30
    input_seq_length = 20
    output_seq_length = 10
    inputs_ids = generate_input_ids(batch_size, input_seq_length)

    network.generate(inputs_ids,
                     max_new_tokens=output_seq_length,
                     do_sample=False,
                     return_dict_in_generate=False)
