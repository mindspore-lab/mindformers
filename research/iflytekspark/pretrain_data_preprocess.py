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
"""Pretrain dataset process script."""
import json
import argparse
import numpy as np
from mindspore.mindrecord import FileWriter
from iflytekspark_tokenizer import IFlytekSparkTokenizer


def process_text(txt):
    """Process raw text."""
    new_txt = txt.strip() \
                 .replace('\\r\\n', '<ret>') \
                 .replace('\\r\n', '<ret>') \
                 .replace('\\n', '<ret>') \
                 .replace('\n', '<ret>')
    return new_txt

def get_text(tokenizer, file_path):
    """Apply template to text and tokenize."""
    prompt_format = (
        "{prompt}<end>"
    )

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            prompt = process_text(item['output'])
            prompt_ids = tokenizer.encode(prompt_format.format_map({'prompt': prompt}))
            input_ids = list(np.array(prompt_ids))
            yield input_ids

def write_mindrecord(tokenizer, data_path, file_name, seq_length=32768):
    """Write mindrecord file."""
    schema = {
        "input_ids": {"type": "int64", "shape": [-1]},
    }
    writer = FileWriter(file_name=file_name, shard_num=1, overwrite=True)
    writer.add_schema(schema)
    writer.open_and_set_header()

    token_all_list = []

    for item in get_text(tokenizer, data_path):
        input_ids = item
        token_all_list.append(input_ids)

    # flatten
    flatten_list = [it for sublist in token_all_list for it in sublist]
    num = int(len(flatten_list) / seq_length)
    redundant = len(flatten_list) % seq_length
    arr_token_flat = np.array(flatten_list[:-redundant])
    arr_token = arr_token_flat.reshape(num, seq_length)

    for item in arr_token:
        sample = {}
        sample['input_ids'] = item
        writer.write_raw_data([sample])

    writer.commit()
    print("Transformation finished! Output file refer: {}".format(file_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', required=True, type=str,
                        help='tokenizer name or path')
    parser.add_argument('--raw_data_path', required=True, type=str,
                        help='Raw data file path.')
    parser.add_argument('--output_filename', default='train_dataset.mindrecord', type=str,
                        help='Output mindrecord file path.')
    parser.add_argument('--seq_length', default=32768, type=int,
                        help='Sequence length of each sample.')

    args = parser.parse_args()
    iflytekspark_tokenizer = IFlytekSparkTokenizer(args.tokenizer)
    raw_data_path = args.raw_data_path
    output_filename = args.output_filename
    write_mindrecord(iflytekspark_tokenizer, raw_data_path, output_filename, args.seq_length+1)
