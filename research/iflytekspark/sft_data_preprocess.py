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
"""Finetune dataset process script."""
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
        "<User> {instruction}<end><Bot> "
    )
    response_format = (
        "{response}<end>"
    )

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            sample = {}
            prompt = process_text(item['instruction']+item['input'])
            prompt_ids = tokenizer.encode(prompt_format.format_map({'instruction': prompt}))
            prompt_len = len(prompt_ids)

            response = process_text(item['output'])
            response_ids = tokenizer.encode(response_format.format_map({'response': response}))
            response_len = len(response_ids)

            input_ids = prompt_ids + response_ids
            loss_mask = [0] * prompt_len + [1] * response_len
            sample['input_ids'] = input_ids
            sample['loss_mask'] = loss_mask

            assert len(input_ids) == len(loss_mask)

            yield sample

def write_mindrecord(tokenizer, data_path, file_name, seq_length=32768, pad_id=0):
    """Write mindrecord file."""
    schema = {
        "input_ids": {"type": "int64", "shape": [-1]},
        "loss_mask": {"type": "int64", "shape": [-1]},
    }
    writer = FileWriter(file_name=file_name, shard_num=1, overwrite=True)
    writer.add_schema(schema)
    writer.open_and_set_header()

    old_input_ids = []
    old_loss_mask = []
    new_sample = {}
    for item in get_text(tokenizer, data_path):
        sample = item
        new_input_ids = sample['input_ids']
        new_loss_mask = sample['loss_mask']
        if len(old_input_ids) + len(new_input_ids) <= seq_length:
            old_input_ids += new_input_ids
            old_loss_mask += new_loss_mask
        else:
            old_loss_mask += [0] * (seq_length - len(old_loss_mask))
            old_input_ids += [pad_id] * (seq_length-len(old_input_ids))

            new_sample['input_ids'] = np.array(old_input_ids)
            new_sample['loss_mask'] = np.array(old_loss_mask)
            writer.write_raw_data([new_sample])
            old_input_ids = new_input_ids
            old_loss_mask = new_loss_mask

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
    parser.add_argument('--pad_id', default=0, type=int,
                        help='Padding token id.')

    args = parser.parse_args()
    iflytekspark_tokenizer = IFlytekSparkTokenizer(args.tokenizer)
    raw_data_path = args.raw_data_path
    output_filename = args.output_filename
    write_mindrecord(iflytekspark_tokenizer, raw_data_path, output_filename, args.seq_length+1, args.pad_id)
