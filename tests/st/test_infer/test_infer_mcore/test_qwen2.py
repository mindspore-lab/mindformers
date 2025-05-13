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
Test module for testing the stand alone Infer interface used for mindformers.
How to run this:
    pytest tests/st/test_infer/test_infer_model/test_standalone_infer.py
"""
import os
import math

import pytest
import jieba
import numpy as np

from mindspore import Model

from mindformers import build_context, MindFormerConfig
from mindformers.experimental.models.qwen2.configuration_qwen2 import Qwen2Config
from mindformers.experimental.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.utils.load_checkpoint_utils import get_load_path_after_hf_convert
from mindformers.modules.transformer import TransformerOpParallelConfig
from research.qwen2.qwen2_tokenizer import Qwen2Tokenizer

from utils import get_qwen_model_config


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


def compare_distance(x1, x2, bench_sim=0.95):
    """compare distance"""
    y1 = list(jieba.cut(x1))
    y2 = list(jieba.cut(x2))
    all_words = _get_all_words(y1, y2)
    laa, lbb = _get_word_vector(y1, y2, all_words)
    sim = _get_calculate_cos(laa, lbb)
    print("calculate sim is:{}".format(str(sim)))
    assert sim >= bench_sim

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_qwen2_0_5b_predict_mcore():
    """
    Feature: Infer interface
    Description: Test mcore interface for prediction.
    Expectation: AssertionError
    """
    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2-0.5B-Instruct/vocab.json"
    merges_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2-0.5B-Instruct/merges.txt"
    load_safetensors = "/home/workspace/mindspore_dataset/weight/Qwen2-0.5B-Instruct"
    seq_length = 128

    config = MindFormerConfig()
    device_num = os.environ.get("DEVICE_NUM")
    config.parallel_config = TransformerOpParallelConfig()
    config.parallel_config.model_parallel = int(device_num) if device_num else 1
    config.use_parallel = bool(device_num)
    config.load_checkpoint = load_safetensors
    config.load_ckpt_format = "safetensors"
    config.auto_trans_ckpt = True
    config.use_legacy = False
    config.run_mode = "predict"
    config.context = MindFormerConfig(mode=0)
    # init context
    build_context(config)
    qwen_model_config = get_qwen_model_config()
    config.model = MindFormerConfig(model_config=qwen_model_config)
    config.model.model_config.parallel_config = config.parallel_config
    config.parallel = MindFormerConfig(parallel_mode="STAND_ALONE")
    model_config = Qwen2Config(**config.model.model_config)

    # build tokenizer
    tokenizer = Qwen2Tokenizer(vocab_file=vocab_file_path, merges_file=merges_file_path)
    # build model
    network = Qwen2ForCausalLM(model_config)
    model = Model(network)
    if config.load_checkpoint:
        config.load_checkpoint = get_load_path_after_hf_convert(config=config, network=network)
        transform_and_load_checkpoint(config, model, network, None, do_predict=True)
    batch_datas = {1: {"prompt": "你好！",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好！"
                                 "<|im_end|>\n<|im_start|>assistant\n你好！有什么可以帮助你的吗？<|im_end|>"},
                   4: {"prompt": "用python编写快速排序",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n用"
                                 "python编写快速排序<|im_end|>\n<|im_start|>assistant\n以下是一个使用Python实现的快速排序"
                                 "算法：\n\n```python\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        "
                                 "return arr\n    else:\n        pivot = arr[0]\n        left = [x for x in arr[1:] "
                                 "if x < pivot]\n        right = [x for x in arr[1:] if x >= pivot]\n        "
                                 "return quick_sort(left) + [pivot] + quick_sort(right)\n\n# 示例输入\narr = [3,6,8,1"},
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
                                   max_length=seq_length,
                                   do_sample=False,
                                   return_dict_in_generate=False)

        for i in range(0, len(outputs)):
            output_text = tokenizer.decode(outputs[i])
            print("test_qwen2_0_5b_predict_standalone, output_text:", output_text)
            compare_distance(output_text, answer)
