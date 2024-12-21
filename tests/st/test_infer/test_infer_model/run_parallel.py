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
Test module for testing the paralleled infer interface used for mindformers.
How to run this:
    pytest tests/st/test_infer/test_infer_model/run_parallel.py
"""
import argparse
import os

import mindspore as ms
from mindspore import Tensor, Model
from mindspore.common import initializer as init

from mindformers import build_context, MindFormerConfig, build_parallel_config, LlamaConfig, \
    LlamaForCausalLM
from mindformers.tools.logger import logger
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.models.glm2 import ChatGLM2Config, ChatGLM2ForConditionalGeneration, ChatGLM3Tokenizer
from research.qwen2.qwen2_tokenizer import Qwen2Tokenizer


def parallel_qwen2_0_5b_predict_mp2():
    """test qwen2-0.5B predict in model_parallel=2 with dynamic shape"""
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "ci_predict_qwen2_0_5b_instruct.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/vocab.json"
    merges_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/merges.txt"
    load_checkpoint = "/home/workspace/mindspore_dataset/weight/ms_safetensor_qwen2_0.5/"

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = True
    config.parallel.strategy_ckpt_config.save_file = "./qwe2_05b_dynamic_ckpt_strategy.ckpt"
    config.load_ckpt_format = "safetensors"
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.load_checkpoint = load_checkpoint
    config.model.model_config.seq_length = seq_length
    config.processor.tokenizer.vocab_file = vocab_file_path
    config.processor.tokenizer.merges_file = merges_file_path

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)

    # build model
    network = LlamaForCausalLM(model_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        batch_size = config.model.model_config.batch_size
        input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    # predict
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
                   8: {"prompt": "I believe the meaning of life is",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nI "
                                 "believe the meaning of life is<|im_end|>\n<|im_start|>assistant\nThe meaning of "
                                 "life is a philosophical question that has been debated for centuries, and there "
                                 "is no one definitive answer to it. Some people believe that the meaning of life "
                                 "is to find happiness and fulfillment in their lives, while others believe that it "
                                 "is to achieve success or recognition.\n\nOthers may argue that the meaning of life "
                                 "is to make a positive impact on the world, to help others, and to contribute to "
                                 "society as a whole. Others may believe that the meaning of life is to pursue "
                                 "knowledge and understanding, to"},
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
            print("parallel_qwen2_0_5b_predict_mp2, output_text:", output_text)
            assert output_text == answer


def parallel_glm3_6b_predict_mp2():
    """test glm3-6B predict in model_parallel=2 with static shape"""
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "ci_predict_glm3_6b.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/chatglm3-6b-tokenizer/tokenizer.model"
    load_checkpoint = "/home/workspace/mindspore_dataset/weight/glm3_6b_ckpt_2cards/"

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = True
    config.parallel.strategy_ckpt_config.save_file = "./glm3_6b_ckpt_strategy.ckpt"
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.parallel_config.vocab_emb_dp = False
    config.load_checkpoint = load_checkpoint
    config.model.model_config.seq_length = seq_length
    config.model.model_config.num_layers = 2
    config.processor.tokenizer.vocab_file = vocab_file_path

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    model_config = ChatGLM2Config(**config.model.model_config)
    model_config.checkpoint_name_or_path = None
    model_name = config.trainer.model_name

    # build tokenizer
    tokenizer = ChatGLM3Tokenizer.from_pretrained(model_name)

    # build model
    network = ChatGLM2ForConditionalGeneration(model_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        batch_size = config.model.model_config.batch_size
        input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    batch_datas = {1: {"prompt": "你好!",
                       "answer": "[gMASK]sop<|user|> \n 你好!<|assistant|>你也可以到解开到一起 schematic彩人生k镜报出的镜 "
                                 "tripters正反馈桌面上的一个大小的跳�勇于对不起ate Motters副 stock光速翻译外als冰盖儿净"
                                 "ting引标 independencehiiterationally�挂erboltene产生了一类� minute Practicality"
                                 "横表现力 st这种类型的faceid从中畔与他们之间positionality田螺的继续ancearrows compatibley"
                                 "先从这里开始了一类ffff提起了和教育化程度椰s脱脱于不顾 pulse开始了一类ffff提起了和教育化"
                                 "程度椰s脱脱于"},
                   4: {"prompt": "用python编写快速排序",
                       "answer": "[gMASK]sop<|user|> \n 用python编写快速排序<|assistant|>感知的储备面 birth方寸间歇models"
                                 "那些符合条件的曲ity专业化程度远 enoughfunding怕ful古今的都是一些列吾速成了一类�干儿 "
                                 "netting away from梦人生解离还是一位堂口englishmen传别 pars许 minimize机遇的都是一些列"
                                 "吾速成了一种方式和安全感的的智慧差oid观音插sitem主人hips replacementute弓遥遥遥遥遥不可"
                                 "避免地txtured品的机会机会 tendencies downfile夹子 netted vacs Spreadsheet字典差法定"
                                 "ities横置交叉得到一个月的longue"},
                   8: {"prompt": "I believe the meaning of life is",
                       "answer": "[gMASK]sop<|user|> \n I believe the meaning of life is<|assistant|>感知的stit "
                                 "glasscapacity够用尽享 convenienceiently enough椰子用尽享 convenienceiently "
                                 "enough椰s脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱"
                                 "脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱脱"
                                 "脱脱脱脱脱脱脱"},
                   }

    for batch_size, batch_data in batch_datas.items():
        input_ids = tokenizer.build_chat_input(batch_data["prompt"], history=[], role='user')[
            "input_ids"]
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
            print("parallel_glm3_6b_predict_mp2, output_text:", output_text)
            assert output_text == answer


def parallel_qwen_moe_predict_mp2():
    """test qwen2-0.5B predict in model_parallel=2 with dynamic shape"""
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "ci_predict_qwen2_57b_a14b_instruct.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/vocab.json"
    merges_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/merges.txt"

    ms.set_seed(123)

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = True
    config.parallel.strategy_ckpt_config.save_file = "./qwen_moe_strategy.ckpt"
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.model.model_config.num_layers = 2
    config.model.model_config.seq_length = seq_length
    config.processor.tokenizer.vocab_file = vocab_file_path
    config.processor.tokenizer.merges_file = merges_file_path

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)

    # build model
    network = LlamaForCausalLM(model_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        batch_size = config.model.model_config.batch_size
        input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    # predict
    batch_datas = {1: {"prompt": "你好！",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好！"
                                 "<|im_end|>\n<|im_start|>assistant\n静脉静脉静脉静脉静脉静脉静脉静脉静脉静脉静脉静脉静脉"
                                 "静脉静脉静脉静脉静脉axteraxteraxteraxteraxteraxteraxteraxteraxteraxteraxteraxteraxte"
                                 "raxteraxter yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet"
                                 " yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet"
                                 " yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet"
                                 " yet yet yet yet yet yet yet yet yet yet yet yet yet yet"},
                   4: {"prompt": "用python编写快速排序",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n用"
                                 "python编写快速排序<|im_end|>\n<|im_start|>assistant\n_time_time_time_time_time_time_"
                                 "time_time_time��������������������������������������"
                                 "������ sourced sourced sourced sourced sourced sourced sourced sourced sourced "
                                 "sourced sourced sourced sourced sourced sourced sourced sourced sourced sourced "
                                 "sourced sourced sourced sourced sourced sourced sourced sourced sourced sourced "
                                 "sourced sourced sourced sourced sourced sourced sourced sourced sourced sourced "
                                 "sourced sourced sourced sourced sourced sourced sourced sourced sourced sourced "
                                 "sourced sourced"},
                   8: {"prompt": "I believe the meaning of life is",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nI "
                                 "believe the meaning of life is<|im_end|>\n<|im_start|>assistant\n.origin.origin."
                                 "origin.origin.origin.origin.origin.origin.origin.origin.origin.origin.origin.origin"
                                 ".origin.origin.origin.origin.origin.origin.origin.origin理事理事理事理事理事理事理事理"
                                 "事理事理事理事理事理事理事理事理事理事理事.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail"
                                 ".Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail"
                                 ".Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail"
                                 ".Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail"
                                 ".Fail.Fail"},
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
            print("parallel_qwen2_0_5b_predict_mp2, output_text:", output_text)
            print("answer, answer:", answer)
            assert output_text == answer


def parallel_qwen2_0_5b_predict_mp2_static():
    """test qwen2-0.5B predict in model_parallel=2 with static shape"""
    # config.model.model_config.is_dynamic = False
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "ci_predict_qwen2_0_5b_instruct.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/vocab.json"
    merges_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/merges.txt"
    load_checkpoint = "/home/workspace/mindspore_dataset/weight/ms_safetensor_qwen2_0.5/"

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = True
    config.parallel.strategy_ckpt_config.save_file = "./qwe2_05b_static_ckpt_strategy.ckpt"
    config.load_ckpt_format = "safetensors"
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.load_checkpoint = load_checkpoint
    config.model.model_config.seq_length = seq_length
    config.model.model_config.is_dynamic = False
    config.model.model_config.do_sample = False
    config.model.model_config.temperature = 1.0
    config.processor.tokenizer.vocab_file = vocab_file_path
    config.processor.tokenizer.merges_file = merges_file_path

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)

    # build model
    network = LlamaForCausalLM(model_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        batch_size = config.model.model_config.batch_size
        input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    # predict
    batch_datas = {4: {"prompt": "你好!",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                                 "你好!<|im_end|>\n<|im_start|>assistant\n你好！很高兴为你提供帮助。"
                                 "有什么我可以帮助你的吗？<|im_end|>"},
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
            print("parallel_qwen2_0_5b_predict_mp2_static, output_text:", output_text)
            assert output_text == answer


TEST_MAP = {
    'parallel_qwen2_0_5b_predict_mp2': parallel_qwen2_0_5b_predict_mp2,
    'parallel_glm3_6b_predict_mp2': parallel_glm3_6b_predict_mp2,
    'parallel_qwen_moe_predict_mp2': parallel_qwen_moe_predict_mp2,
    'parallel_qwen2_0_5b_predict_mp2_static': parallel_qwen2_0_5b_predict_mp2_static,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of llama2 model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
