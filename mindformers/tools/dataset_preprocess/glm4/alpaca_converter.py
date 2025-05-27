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
fastchat stanford alpaca data convert tools.
"""
import argparse
import json
import pathlib
import os
from tqdm import tqdm


def main(data_path, output_path):
    data_path = pathlib.Path(data_path)
    with data_path.open() as f:
        data = json.load(f)

    sources = []
    total_example = len(data)
    with tqdm(total=total_example, desc="Getting input", unit="row") as pbar:
        for example in data:
            if example.get("input", "") == "":
                sources.append(example['instruction'])
            else:
                instruction = example['instruction']
                if instruction[-1] == ".":
                    instruction = instruction[:-1]
                instruction = instruction + ": " + example['input']
                sources.append(instruction)
            pbar.update(1)

    targets = []
    with tqdm(total=total_example, desc="Getting output", unit="row") as pbar:
        for example in data:
            targets.append(example['output'])
            pbar.update(1)

    new_data = []
    total_sources = len(sources)
    with tqdm(total=total_sources, desc="Appending messages", unit="row") as pbar:
        for s, t in zip(sources, targets):
            new_data.append({
                "messages": [
                    {
                        "role": "user",
                        "content": s,
                    },
                    {
                        "role": "assistant",
                        "content": t,
                    },
                ]
            })
            pbar.update(1)

    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(output_path, flags_, 0o750), 'w', encoding='utf-8') as f:
        total_data = len(new_data)
        with tqdm(total=total_data, desc="Saving json files", unit="row") as pbar:
            for data in new_data:
                f.write(json.dumps(data) + "\n")
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="alpaca_data.json")
    parser.add_argument("--output_path", type=str, default="alpaca_glm4_data.jsonl")
    args = parser.parse_args()
    main(args.data_path, args.output_path)
