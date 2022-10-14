# Copyright 2022 Huawei Technologies Co., Ltd
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
Converting the WMT datasets.
"""

import argparse
from tqdm import tqdm
import os
import numpy as np

from mindspore.mindrecord import FileWriter
from transformer.models.t5.T5Tokenizer import T5Tokenzier


def read_text(train_file):
    """Read the text files and return a list."""
    with open(train_file) as fp:
        data = []
        for line in fp:
            line = line.strip()
            if line:
                data.append(line)
    return data


def convert_to_json(source, target):
    """Convert text source and target language into the dict and return an iterator."""
    for src, tgt in zip(source, target):
        yield {'en': src, 'ro': tgt}


def pad_max_length(text, pad_id, pad_length):
    """Pad the text to the max_length with given the pad id."""
    if len(text) < pad_length:
        text = text + [pad_id] * (pad_length - len(text))
    else:
        text = text[:pad_length]
    return text


def read_and_convert(split,
                     data_path,
                     tokenizer,
                     output_file_name,
                     src_pad_length=1024,
                     tgt_pad_length=128):
    """Read the arguments and convert to the mindrecords."""
    source_paht = os.path.join(data_path, f'{split}.source')
    src = read_text(source_paht)
    target_source = os.path.join(data_path, f'{split}.target')
    tgt = read_text(target_source)

    iterator = convert_to_json(src, tgt)
    pad_id = tokenizer.s.pad_id()

    writer = FileWriter(file_name=output_file_name, shard_num=1, overwrite=True)
    writer.add_schema({"input_ids": {"type": "int32", "shape": [-1]},
                       "attention_mask": {"type": "float32", "shape": [-1]},
                       "labels": {"type": "int32", "shape": [-1]}}, "dataset_schema")

    for item in tqdm(iterator, desc="Converting to MindRecords", total=len(src)):
        src = tokenizer.convert_str_to_ids(item['en'])
        src_attention_mask = [1.0] * len(src)
        src = pad_max_length(src, pad_id, src_pad_length)
        src_attention_mask = pad_max_length(src_attention_mask, 0, src_pad_length)
        tgt = tokenizer.convert_str_to_ids(item['ro'])
        tgt = pad_max_length(tgt, pad_id, tgt_pad_length)

        new_item = {'input_ids': np.array(src).astype(np.int32),
                    "attention_mask": np.array(src_attention_mask).astype(np.float32),
                    "labels": np.array(tgt).astype(np.int32)}
        writer.write_raw_data([new_item])
    writer.commit()
    print(f'MindRecords are written to {output_file_name}, total transformed {len(src)} examples.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="read dataset and save it to minddata")
    parser.add_argument("--split", type=str, default='train', choices=['train', 'test', 'val'],
                        help="The dataset split name to be preprocessed.")
    parser.add_argument("--sp_model_path", type=str, default=None, required=True,
                        help="The sentence piece model file path")
    parser.add_argument("--output_file_path", type=str, default=None, required=True,
                        help="The output directory path of mindspore records.")
    parser.add_argument("--raw_dataset", type=str, default=None, required=True,
                        help="The raw WMT16 dataset file directory.")
    parser.add_argument("--input_ids_length", type=int, default=1024,
                        help="The maximum total input ids length.")
    parser.add_argument("--labels_length", type=int, default=128,
                        help="The maximum total labels length.")
    args_opt = parser.parse_args()

    token = T5Tokenzier(sp_model=args_opt.sp_model_path)
    if not os.path.exists(args_opt.output_file_path):
        os.makedirs(args_opt.output_file_path, exist_ok=True)
    output_path = os.path.join(args_opt.output_file_path, f'{args_opt.split}_mindrecord')
    read_and_convert(split=args_opt.split,
                     data_path=args_opt.raw_dataset,
                     tokenizer=token,
                     src_pad_length=args_opt.input_ids_length,
                     tgt_pad_length=args_opt.labels_length,
                     output_file_name=output_path)
