# -*-coding:utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd
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
"""eval longbench generate method"""
import os
import argparse
import json
from tqdm import tqdm

from mindspore import context
from mindformers import TransformerOpParallelConfig, MindFormerConfig, init_context

from glm32k_config import ChatGLM32kConfig
from glm32k import ChatGLM32kForConditionalGeneration
from glm32k_tokenizer import ChatGLM32kTokenizer

context.set_context(mode=0, device_target="Ascend")

DATASET_PROMPT = "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答："


def read_json_file(dataset_file):
    r"""
    Read original dataset

    Args:
       dataset_file (str): the dataset file.
    """
    raw_data = []
    for line in open(dataset_file, 'r'):
        raw_data.append(json.loads(line))
    return raw_data


def load_model_and_tokenizer(args_para):
    r"""
    Load glm3-32k model and tokenizer

    Args:
       args_para: input parameters
    """

    config = MindFormerConfig(os.path.realpath(args_para.config_path))
    config.use_parallel = args_para.use_parallel

    if not config.use_parallel:
        config.context.device_id = args_para.device_id

    init_context(use_parallel=config.use_parallel,
                 context_config=config.context,
                 parallel_config=config.parallel)

    glm32k_config = ChatGLM32kConfig(**config.model.model_config)
    glm32k_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)

    if args_para.checkpoint_path and not config.use_parallel:
        glm32k_config.checkpoint_name_or_path = args_para.checkpoint_path

    print("starting ......")
    tokenizer = ChatGLM32kTokenizer(config.processor.tokenizer.vocab_file)
    model = ChatGLM32kForConditionalGeneration(glm32k_config)
    model.set_train(False)
    return model, tokenizer


def get_pred(data, args_para, out_path):
    r"""
    get the generated result and save the result to the file
    """
    glm32k_model, glm32k_tokenizer = load_model_and_tokenizer(args_para)
    for json_obj in tqdm(data):
        prompt = DATASET_PROMPT.format(**json_obj)
        tokenized_prompt = \
            glm32k_tokenizer(prompt, truncation=False, return_tensors="ms", add_special_tokens=False).input_ids

        if len(tokenized_prompt) > args_para.max_length:
            half = int(args_para.max_length / 2)
            prompt = glm32k_tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                     glm32k_tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        input_value = glm32k_tokenizer.build_chat_input(prompt)

        context_length = input_value["input_ids"].shape[-1]
        output = glm32k_model.generate(input_ids=input_value["input_ids"],
                                       max_new_tokens=args_para.max_gen,
                                       num_beams=1,
                                       use_past=args_para.use_past,
                                       do_sample=False,
                                       temperature=1.0)[0]

        pred_value = glm32k_tokenizer.decode(output[context_length:], skip_special_tokens=True)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred_value, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],
                       "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='run_glm32k.yaml', type=str,
                        help='config')
    parser.add_argument('--max_length', default=31500, type=int,
                        help='max length')
    parser.add_argument('--max_gen', default=128, type=int,
                        help='max new tokens')
    parser.add_argument('--device_id', default=5, type=int,
                        help='device_id')
    parser.add_argument('--use_past', action="store_true",
                        help="use past")
    parser.add_argument('--use_parallel', default=False, type=bool,
                        help="use parallel")
    parser.add_argument('--checkpoint_path', default='/path/mindspore_models/glm32k.ckpt', type=str,
                        help="checkpoint_path")
    parser.add_argument('--input_dataset_file', default='/path/eval_dataset/dureader.jsonl', type=str,
                        help="input dataset file")
    parser.add_argument('--output_file', default=f"pred", type=str,
                        help="output file")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=25, help="")

    opt_para = parser.parse_args()

    raw_datasets = read_json_file(opt_para.input_dataset_file)

    # choose the datasets according to [start_index: end_index]
    sorted_ids = sorted([per_data["_id"] for per_data in raw_datasets])[opt_para.start_index: opt_para.end_index]
    data_subsets = []
    for cur_data in raw_datasets:
        for cur_id in sorted_ids:
            if cur_id == cur_data["_id"]:
                data_subsets.append(cur_data)

    output_file = opt_para.output_file + '/{}.jsonl'.format(opt_para.start_index)
    get_pred(data_subsets, opt_para, output_file)
