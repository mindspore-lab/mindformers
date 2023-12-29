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
"""mbpp evaluate using mindspore lite"""

import json
import argparse
from tqdm import tqdm
import mindspore as ms
from mindformers.pipeline import pipeline

from wizardcoder_tokenizer import WizardCoderTokenizer


def read_mbpp(path):
    """read mbpp file"""
    mbpp_problems = {}
    with open(path, "r", encoding="utf-8") as in_file:
        for line in in_file:
            item = json.loads(line.strip())
            mbpp_problems[item["task_id"]] = item
    return mbpp_problems


def generate_prompt(input_problem):
    """construct the prompt"""
    query_prompt = \
        f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a Python script for this problem:
{input_problem}

### Response:"""
    return query_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch_size')
    parser.add_argument('--seq_length', default=2048, type=int,
                        help='batch_size')
    parser.add_argument('--tokenizer_path', default='/path/mindspore_models/', type=str,
                        help='tokenizer_path')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--start_index', default=0, type=int,
                        help='start_index')
    parser.add_argument('--end_index', default=0, type=int,
                        help='end_index')
    parser.add_argument('--output_path', default="", type=str,
                        help='output_path')
    parser.add_argument('--mbpp_path', default="", type=str,
                        help='mbpp path')
    args = parser.parse_args()

    print("device id: ", args.device_id)
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)
    tokenizer = WizardCoderTokenizer(
        vocab_file=args.tokenizer_path + "vocab.json",
        merge_file=args.tokenizer_path + "merges.txt"
    )
    model_path = (f"wizardcoder-15b_mslite_inc/prefill_seq{args.seq_length}_bs{args.batch_size}_graph.mindir",
                  f"wizardcoder-15b_mslite_inc/decode_seq{args.seq_length}_bs{args.batch_size}_graph.mindir")
    ge_config_path = "lite.ini"
    pipeline_task = pipeline(task="text_generation", model=model_path, backend="mslite", tokenizer=tokenizer,
                             ge_config_path=ge_config_path, model_type="mindir", infer_seq_length=args.seq_length,
                             add_special_tokens=False,
                             device_id=args.device_id)

    print("pipeline create success!")
    problems = read_mbpp(args.mbpp_path)
    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = []
    for task_id in task_ids:
        prompt = f"\n{problems[task_id]['text']}\nTest examples:"
        if task_id == 493:
            # The test examples are too long. We choose to only include the function name.
            test_example = problems[task_id]['test_list'][0]
            prompt += f"\ncalculate_polygons(startx, starty, endx, endy, radius)"
        else:
            for test_example in problems[task_id]['test_list']:
                prompt += f"\n{test_example}"
        prompts.append(prompt)
    num_samples = len(prompts)
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.json'.format(args.start_index + i)

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_prompt(prompt)]
        print(prompt_batch)
        print(len(prompt_batch))
        output = pipeline_task(prompt_batch,
                               do_sample=False,
                               max_length=2048,
                               eos_token_id=0,
                               pad_token_id=49152,
                               skip_special_tokens=True)
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump({"output": output}, f)
