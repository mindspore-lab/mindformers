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
"""generate mindrecord script"""
import os
import argparse
import collections
from random import shuffle
import datasets
import numpy as np
from tqdm import tqdm
from mindspore.mindrecord import FileWriter
from telechat_tokenizer import TelechatTokenizer

class TelechatDataset:
    """TelechatDataset"""
    def __init__(self, output_path, seed, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.raw_datasets = datasets.load_dataset(path="json", data_files=dataset_name)

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_prompt(self, sample):
        return "<_user>" + sample['input'] + "<_bot>"

    def get_prompt_and_answer(self, sample):
        return "<_user>" + sample['input'] + "<_bot>" + sample['output'] + "<_end>"


def write_instance_to_file(writer, instance):
    """write the instance to file"""
    input_ids = instance["input_ids"]
    labels = instance["labels"]

    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids).astype(np.int32)
    features["labels"] = np.asarray(labels).astype(np.int32)
    writer.write_raw_data([features])
    return features


def make_input_mask(labels, tokenizer):
    """generate input mask"""
    user_token_id = tokenizer.convert_tokens_to_ids(args.user_token)
    bot_token_id = tokenizer.convert_tokens_to_ids(args.bot_token)
    end_token_id = tokenizer.convert_tokens_to_ids(args.end_token)
    target_labels = np.zeros((1, args.max_length))
    indices_user = np.where(np.array(labels) == user_token_id)[0]
    indices_bot = np.where(np.array(labels) == bot_token_id)[0]
    indices_end = np.where(np.array(labels) == end_token_id)[0]
    assert len(indices_user) == len(indices_bot) == len(indices_end)
    for i in range(len(indices_bot)):
        user_idx = indices_user[i]
        bot_idx = indices_bot[i]
        end_idx = indices_end[i]
        target_labels[0][bot_idx:end_idx + 1] = 1
        target_labels[0][user_idx] = 1
    return target_labels


def process_dataset(current_dataset, tokenizer, max_seq_len):
    """process dataset."""
    dataset = []
    all_lines = []
    for _, tmp_data in enumerate(current_dataset):
        input_data = tmp_data['input']
        if not input_data.startswith("<_user>"):
            input_data = "<_user>" + input_data
        output = tmp_data['output']
        if "<_bot>" in input_data: ### multiturn
            concat_line = ""
            input_turns = input_data.split("<_user>")[1:]
            for item in input_turns:
                if "<_bot>" in item:
                    concat_line += "<_user>" + item + "<_end>"
                else:
                    concat_line += "<_user>" + item + "<_bot>"
            concat_line += output + "<_end>"
        else: ####single turn
            concat_line = str(input_data) + "<_bot>" + str(output) + "<_end>"
        assert concat_line.count("<_user>") == concat_line.count("<_bot>") == concat_line.count("<_end>")
        all_lines.append(concat_line)
    shuffle(all_lines)
    previous_corpus_token_cnt = 0
    shard = []
    padding_out = []
    for corpus in tqdm(all_lines):
        corpus_ids = tokenizer(corpus)
        if previous_corpus_token_cnt + len(corpus_ids["input_ids"]) < max_seq_len:
            shard.append(corpus)
            previous_corpus_token_cnt += len(corpus_ids["input_ids"])
        else:
            shard_output = "".join(shard)
            shard_output = (args.max_length - previous_corpus_token_cnt) * tokenizer.pad_token + shard_output
            assert len(tokenizer(shard_output)["input_ids"]) == max_seq_len
            if shard_output.count("<_user>") >= 1:
                padding_out.append(shard_output)
            if len(corpus_ids["input_ids"]) < max_seq_len:
                shard = [corpus]
                previous_corpus_token_cnt = len(corpus_ids["input_ids"])
            else:
                shard = []
                previous_corpus_token_cnt = 0
    print("prompt length: ", len(padding_out))
    for dt in padding_out:
        tokens = tokenizer(dt)
        tokens['labels'] = make_input_mask(tokens["input_ids"], tokenizer)
        dataset.append(tokens)
    return dataset


def make_dataset():
    """make dataset."""
    raw_dataset = TelechatDataset(args.output_path, args.seed, args.input_dataset_file)
    train_dataset = raw_dataset.get_train_data()
    tokenizer = TelechatTokenizer(args.vocab_file_path, fast_tokenizer=True,
                                  trust_remote_code=True, padding_side="left")
    train_dataset = process_dataset(train_dataset, tokenizer, args.max_length)
    print("***** Writing to output files *****")
    print("Output File: %s", args.output_dataset_file)
    writer = FileWriter(args.output_dataset_file, 1)
    data_schema = {"input_ids": {"type": "int32", "shape": [-1]},
                   "labels": {"type": "int32", "shape": [-1]}}
    writer.add_schema(data_schema, "lm-schema")
    for dataset in tqdm(train_dataset):
        instance = {"input_ids": dataset["input_ids"], "labels": dataset["labels"]}
        write_instance_to_file(writer, instance=instance)
    writer.commit()
    print(">>>> Transform dataset finished <<<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_file", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument('--vocab_file_path', default='', type=str, help='which model to use.')
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1233)
    parser.add_argument("--user_token", type=str, default="<_user>", help="user token")
    parser.add_argument("--bot_token", type=str, default="<_bot>", help="bot token")
    parser.add_argument("--end_token", type=str, default="<_end>", help="end token")
    args = parser.parse_args()

    if args.output_path and not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    args.output_dataset_file = os.path.join(args.output_path, "new_dataset.mindrecord")
    make_dataset()
