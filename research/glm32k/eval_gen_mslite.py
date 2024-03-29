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
"""eval longbench generate method by lite."""
import argparse
import json

# pylint: disable=W0611
import mindspore_lite as mslite

from mindformers.tools.utils import str2bool
from mindformers.models import ChatGLM3Tokenizer
from mindformers.inference import InferConfig, InferTask

DATASET_PROMPT = "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答："


def read_jsonl_file(dataset_file):
    r"""
    Read original dataset
    Args:
       dataset_file (str): the dataset file.
    """
    raw_data = []
    for line in open(dataset_file, 'r'):
        raw_data.append(json.loads(line))
    return raw_data


def create_lite_model(arg, tokenizer):
    """build infer pipeline for infer config."""
    lite_config = InferConfig(
        prefill_model_path=arg.prefill_model_path,
        increment_model_path=arg.increment_model_path,
        model_type="mindir",
        model_name=arg.model_name,
        ge_config_path=arg.config_path,
        device_id=arg.device_id,
        infer_seq_length=arg.seq_length,
        dynamic=arg.dynamic,
        paged_attention=arg.paged_attention,
        pa_block_size=arg.pa_block_size,
        pa_num_blocks=arg.pa_num_blocks,
    )
    lite_infer_model = InferTask.get_infer_task("text_generation", lite_config, tokenizer=tokenizer)
    return lite_infer_model


# pylint: disable=W0702
def get_pred(opts, datasets, output_path):
    """lite infer main."""
    tokenizer = ChatGLM3Tokenizer(opts.tokenizer_path)
    lite_infer_model = create_lite_model(opts, tokenizer)
    for json_obj in datasets:
        user_input = DATASET_PROMPT.format(**json_obj)
        print(f'user_input is {[user_input]}')
        output = lite_infer_model.infer(user_input,
                                        do_sample=opts.do_sample,
                                        top_k=opts.top_k,
                                        top_p=opts.top_p,
                                        repetition_penalty=opts.repetition_penalty,
                                        temperature=opts.temperature,
                                        max_length=opts.max_length,
                                        max_new_tokens=opts.max_output_length,
                                        num_beams=1,
                                        is_sample_acceleration=opts.is_sample_acceleration,
                                        add_special_tokens=opts.add_special_tokens)
        print(f'output is {output}')
        try:
            pred = [(output[0].split('回答：<|assistant|>')[1]).strip(' \n ')]
        except:
            pred = output
        print(f'pred is {pred}')
        with open(output_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],
                       "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default=0, type=int,
                        help='ID of the target device')
    parser.add_argument('--model_name', default="common", type=str,
                        help="The model name")
    parser.add_argument('--seq_length', default=2048, type=int,
                        help="This model dir path.")
    parser.add_argument('--tokenizer_path', default=None, type=str,
                        help="Tokenizer model to load.")
    parser.add_argument('--prefill_model_path', default=None, type=str,
                        help="This full model path.")
    parser.add_argument('--increment_model_path', default=None, type=str,
                        help="When use kv-cache, this is cache mode path.")
    parser.add_argument('--config_path', default=None, type=str,
                        help="ge config file path.")
    parser.add_argument('--do_sample', default=False, type=str2bool,
                        help="Whether postprocess in graph or not.")
    parser.add_argument('--top_k', default=1, type=int,
                        help="top k.")
    parser.add_argument('--top_p', default=1.0, type=float,
                        help="top p.")
    parser.add_argument('--repetition_penalty', default=1.0, type=float,
                        help="repetition penalty.")
    parser.add_argument('--temperature', default=1.0, type=float,
                        help="The value used to modulate the next token probabilities.")
    parser.add_argument('--max_length', default=512, type=int,
                        help="The maximum word length that can be generated.")
    parser.add_argument('--max_output_length', default=128, type=int,
                        help="The maximum output length that can be generated.")
    parser.add_argument('--is_sample_acceleration', default=False, type=str2bool,
                        help="Whether postprocess in graph or not.")
    parser.add_argument('--add_special_tokens', default=False, type=str2bool,
                        help="Whether preprocess add special tokens or not.")
    parser.add_argument('--dynamic', default=False, type=str2bool,
                        help="Whether use dynamic inference.")
    parser.add_argument('--paged_attention', default=False, type=str2bool,
                        help="Whether use paged attention.")
    parser.add_argument('--pa_block_size', default=16, type=int,
                        help="Block size of paged attention.")
    parser.add_argument('--pa_num_blocks', default=512, type=int,
                        help="The number of blocks of paged attention.")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="batch size for inference data")
    parser.add_argument('--input_dataset_file', default='/path/eval_dataset/dureader.jsonl', type=str,
                        help="input dataset file")
    parser.add_argument('--output_file', default=f"pred", type=str,
                        help="output file")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=25, help="")

    args = parser.parse_args()
    if len(args.config_path.split(',')) > 1:
        args.config_path = args.config_path.split(',')

    raw_datasets = read_jsonl_file(args.input_dataset_file)
    sorted_ids = sorted([per_data["_id"] for per_data in raw_datasets])[args.start_index: args.end_index]
    data_subsets = []
    for cur_data in raw_datasets:
        for cur_id in sorted_ids:
            if cur_id == cur_data["_id"]:
                data_subsets.append(cur_data)
    output_file = args.output_file + '/{}.jsonl'.format(args.start_index)
    get_pred(args, data_subsets, output_file)
