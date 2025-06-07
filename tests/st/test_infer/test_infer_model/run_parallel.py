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
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, Model, set_context
from mindspore.common import initializer
from mindspore.communication import init

from mindformers import build_context, MindFormerConfig, build_parallel_config, LlamaConfig, \
    LlamaForCausalLM, Trainer
from mindformers.tools.logger import logger
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from research.deepseek3.moe import SharedParallelMLP
from research.qwen2.qwen2_tokenizer import Qwen2Tokenizer
from research.deepseek3.deepseek3_config import DeepseekV3Config
from research.telechat2.telechat_config import TelechatConfig
from research.telechat2.telechat_tokenizer import TelechatTokenizer
from research.telechat2.infer.telechat import ParallelTelechatForCausalLM


def get_config():
    """get config of testcase"""
    base_config = DeepseekV3Config(
        param_init_dtype=mstype.bfloat16,
        compute_dtype=mstype.bfloat16,
        use_past=True,
        num_heads=16,
        hidden_size=1024,
        use_flash_attention=True,
        qkv_has_bias=False,
        rotary_dtype=mstype.bfloat16,
        num_blocks=16,
        block_size=256,
        out_proj_has_bias=False,
        vocab_size=1000,
        num_layers=2,
        seq_length=512,
        mlp_has_bias=False,
        ffn_concat=True,
        intermediate_size=4096,
    )
    parallel_config = MindFormerConfig(
        tensor_parallel=2,
        context_parallel=1,
        vocab_emb_dp=False
    )
    base_config.parallel_config = parallel_config
    return base_config


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
    config.output_dir = './qwen2_05b_dynamic_output'
    config.parallel.strategy_ckpt_save_file = "./qwen2_05b_dynamic_ckpt_strategy.ckpt"
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
        input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=initializer.One())
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
    config.parallel.strategy_ckpt_save_file = "./glm3_6b_ckpt_strategy.ckpt"
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.parallel_config.vocab_emb_dp = False
    config.load_checkpoint = load_checkpoint
    config.model.model_config.seq_length = seq_length
    config.processor.tokenizer.vocab_file = vocab_file_path

    # init context
    build_context(config)
    task_trainer = Trainer(config)

    batch_datas = {1: {"prompt": "你好!",
                       "answer": "[gMASK]sop 你好!我是人工智能助手。很高兴能为你提供帮助。请问有什么问题我可以帮你解答吗?"},
                   4: {"prompt": "用python编写快速排序",
                       "answer": "[gMASK]sop 用python编写快速排序算法\n\n 快速排序是一种高效的排序算法，其基本思想是通过"
                                 "一趟排序将待排序的数据分割成独立的两部分，其中一部分的所有数据都比另一部分的所有数据要小，"
                                 "然后再按此方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，以此达到整个数据"
                                 "变成有序序列。\n\n下面是使用 Python 编写快速排序算法的示例代码：\n\n```python\ndef "
                                 "quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr["},
                   8: {"prompt": "I believe the meaning of life is",
                       "answer": "[gMASK]sop I believe the meaning of life is to seek knowledge and understanding. "
                                 "What are some specific ways to achieve this?\nThere are many ways to achieve the goal"
                                 " of seeking knowledge and understanding. Here are a few specific suggestions:\n\n1. "
                                 "Read widely: One of the best ways to gain knowledge and understanding is to read "
                                 "widely in a variety of subjects. This can help you develop a broad range of knowledge"
                                 " and perspectives.\n2. Ask questions: Seeking knowledge and understanding often "
                                 "involves asking questions. Don't be afraid to ask questions of others, and be sure "
                                 "to ask yourself questions as well.\n3. Engage"},
                   }

    for batch_size, batch_data in batch_datas.items():
        answer = batch_data["answer"]
        input_data = batch_data["prompt"]
        input_data_list = []
        for i in range(0, batch_size):
            input_data_list.append(input_data)
        outputs = task_trainer.predict(predict_checkpoint=config.load_checkpoint,
                                       input_data=input_data_list,
                                       max_length=seq_length,
                                       batch_size=batch_size)
        for i in range(0, len(outputs)):
            output_text = outputs[i]
            print("parallel_glm3_6b_predict_mp2, output_text:", output_text['text_generation_text'][0])
            assert output_text['text_generation_text'][0] == answer


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
    config.parallel.strategy_ckpt_save_file = "./qwen_moe_strategy.ckpt"
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
        input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=initializer.One())
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
    config.output_dir = './qwen2_05b_static_output'
    config.parallel.strategy_ckpt_save_file = "./qwen2_05b_static_ckpt_strategy.ckpt"
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


def parallel_shared_expert_predict_mp2():
    """test shared expert with tensor model parallel size 2 """
    jit_level = "O0"
    infer_boost = "on"
    set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": jit_level, "infer_boost": infer_boost})

    # init communication
    init()
    initialize_model_parallel(tensor_model_parallel_size=2)

    base_config = get_config()
    net = SharedParallelMLP(base_config, base_config.intermediate_size)

    bs = 2
    seq_len = base_config.seq_length
    hidden_size = base_config.hidden_size
    x = Tensor(np.random.rand(bs, seq_len, hidden_size)).astype(mstype.bfloat16)

    output = net(x)
    assert output.shape == (bs, seq_len, hidden_size)


def parallel_telechat2_predict_mp2():
    """test telechat2 predict in model_parallel=2 with dynamic shape"""
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "ci_predict_telechat2_parallel.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Telechat2-tokenizer/tokenizer.model"

    ms.set_seed(123)

    seq_length = 128
    max_new_tokens = 8
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = True
    config.output_dir = './telechat2_dynamic_output'
    config.parallel.strategy_ckpt_save_file = "./telechat2_dynamic_ckpt_strategy.ckpt"
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.model.model_config.seq_length = seq_length
    config.processor.tokenizer.vocab_file = vocab_file_path

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    model_config = TelechatConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    chat_template = "{%- if tools %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{-'<_system>'+messages[0]['content'] }}\n    {%- else %}\n        {{- '<_system>'+'你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。' }}\n    {%- endif %}\n    {{- '\\n\\n# 可用工具\\n你可以调用<tools></tools>标签中包含的一个或多个工具来辅助你回答问题,以下是可用工具详情：\\n<tools>\\n' }}\n    {%- for tool in tools %}\n        {{- tool | tojson }}\n        {{-'\\n'}}\n    {%- endfor %}\n    {{- '</tools>\\n\\n# 调用方法\\n你需要遵循工具的要求，使用json格式返回工具名称及参数，并用<tool_call></tool_call>包含。下方是一个调用模板：\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call>\\n\\n' }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<_system>' + messages[0]['content'] + '\\n' }}\n    {%- else %}\n        {{- '<_system>'+'你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == 'user') %}\n        {{- '<_user>' + message.content }}\n    {%- elif message.role == 'bot' or message.role == 'assistant' %}\n        {{- '<_bot>' }}\n        {%- if message.content %}\n            {{- message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {%- if loop.index0 == 0 %}\n                {{-'<tool_call>'}}\n            {%- else %}\n                {{-'\\n<tool_call>'}}\n            {%- endif %}\n            {{- '\\n{\"name\": \"' }}{{ tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<_end>\\n' }}\n    {%- elif message.role == 'tool' %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != 'tool') %}\n            {{- '<_user>'+'<tool_response>\\n' }}\n        {%- else %}\n            {{- '\\n<tool_response>\\n' }}\n        {%- endif %}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<_bot>' }}\n{%- endif %}"
    tokenizer = TelechatTokenizer(vocab_file=vocab_file_path, chat_template=chat_template, fast_tokenizer=True)

    # build model
    network = ParallelTelechatForCausalLM(model_config)

    # predict
    batch_datas = {1: {"prompt": "你好！",
                       "answer": "stoutFramesInstallepoch概要 americ鼎环城"},
                   4: {"prompt": "用python编写快速排序",
                       "answer": "多姿 dbc Kids teasing怕我作出了 母亲 养"},
                   8: {"prompt": "I believe the meaning of life is",
                       "answer": "ListView annotated莆田 Vanity积累的 Hispanic conceived反思"}}
    for batch_size, batch_data in batch_datas.items():
        messages = [{"role": "user", "content": batch_data["prompt"]}]
        inputs_chat = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(inputs_chat)["input_ids"]

        input_ids_list = []
        answer = batch_data["answer"]
        for i in range(0, batch_size):
            input_ids_list.append(input_ids)
        outputs = network.generate(input_ids_list,
                                   max_length=seq_length,
                                   max_new_tokens=max_new_tokens,
                                   do_sample=True,
                                   return_dict_in_generate=False)

        for i in range(0, len(outputs)):
            output = outputs[i][len(input_ids_list[i]):]
            output_text = tokenizer.decode(output)
            print("parallel_telechat2_predict_mp2, output_text:", output_text)
            if not i:
                assert output_text == answer


TEST_MAP = {
    'parallel_qwen2_0_5b_predict_mp2': parallel_qwen2_0_5b_predict_mp2,
    'parallel_glm3_6b_predict_mp2': parallel_glm3_6b_predict_mp2,
    'parallel_qwen_moe_predict_mp2': parallel_qwen_moe_predict_mp2,
    'parallel_qwen2_0_5b_predict_mp2_static': parallel_qwen2_0_5b_predict_mp2_static,
    'parallel_shared_expert_predict_mp2': parallel_shared_expert_predict_mp2,
    'parallel_telechat2_predict_mp2': parallel_telechat2_predict_mp2,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of llama2 model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
