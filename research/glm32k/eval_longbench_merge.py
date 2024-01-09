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
"""eval longbench merge method"""
import os
import json
import argparse


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


def merge_result(args):
    r"""
    merge all results to a single file

    Args:
       args: input parameters
    """
    all_unmerged_files = os.listdir(args.need_merge_path)
    final_file = os.path.join(args.merged_path, "dureader.jsonl")
    for file in all_unmerged_files:
        cur_path = os.path.join(args.need_merge_path, file)
        for line in open(cur_path, 'r'):
            cur_data = json.loads(line)
            with open(final_file, "a", encoding="utf-8") as f:
                json.dump(cur_data, f, ensure_ascii=False)
                f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--need_merge_path',
                        default='/path/pred/',
                        type=str, help="Original files")
    parser.add_argument('--merged_path',
                        default='/path/merged/',
                        type=str, help="The final merged path")

    opt_para = parser.parse_args()

    if not os.path.exists(opt_para.merged_path):
        os.makedirs(opt_para.merged_path)

    merge_result(opt_para)
