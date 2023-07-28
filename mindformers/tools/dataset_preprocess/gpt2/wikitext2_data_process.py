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

"""
Data process for wikitext-2 Dataset

The generated dataset has output columns: ["input_ids", "attention_mask", "labels"],
and the data type of three columns is int64.

Columns：
    input_ids: the tokenized inputs, Tensor of shape :math:`(batch, seq_length)`.
    attention_mask: the mask indicating whether each position is a valid input and is not the added prompt,
                    Tensor of shape :math:`(batch, seq_length)`.
    labels: same as input_ids, Tensor of shape :math:`(batch, seq_length)`.

About wikitext-2 Dataset:
    The wikitext-2 language modeling dataset is a collection of over 100 million tokens extracted from the set
    of verified Good and Featured articles on Wikipedia. As it is composed of full articles, the dataset is
    well suited for models that can take advantage of long term dependencies.

    You can unzip the dataset files into the following structure:

    .. code-block::

        .
        └── wikitext-2
             ├── wiki.test.tokens
             ├── wiki.train.tokens
             └── wiki.valid.tokens

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import numpy as np

from mindspore.mindrecord import FileWriter
from mindformers.auto_class import AutoTokenizer


def create_instance(tokenizer, sentence, ids, max_length=None):
    """A single sample instance for LM task."""

    pair_ids = None
    if len(sentence) == 2:
        pair_ids = tokenizer.encode(sentence[1])

    output = tokenizer.prepare_for_model(ids=ids,
                                         pair_ids=pair_ids,
                                         add_special_tokens=True,
                                         max_length=max_length,
                                         padding='max_length',
                                         truncate_direction="LEFT",
                                         return_overflowing_tokens=False,
                                         return_attention_mask=True)
    return output


def write_instance_to_file(writer, instance):
    """write the instance to file"""
    input_ids = instance["input_ids"]
    attention_mask = instance["attention_mask"]
    labels = instance["input_ids"]

    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids)
    features["attention_mask"] = np.asarray(attention_mask)
    features["labels"] = np.asarray(labels)
    writer.write_raw_data([features])

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../wikitext-2/wiki.valid.tokens",
                        help='Input raw text file. ')
    parser.add_argument("--output_file", type=str, default="../wikitext2_processed/wikitext-2.mindrecord",
                        help='Output MindRecord file. ')
    parser.add_argument("--num_splits", type=int, default=1,
                        help="The MindRecord file will be split into the number of partition. ")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length. ")
    parser.add_argument("--tokenizer_type", type=str, default="gpt2",
                        help="Tokenizer type, can be set to any tokenizer "
                             "if its relevant model supports prompt text classification. ")
    parser.add_argument("--data_columns", type=list, default=["input_ids", "attention_mask", "labels"],
                        help="The data columns which should be saved in mindrecord. This can refer used yaml file. ")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_type)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens([])

    input_file = args.input_file
    logging.info("***** Reading from input files *****")
    logging.info("Input File: %s", input_file)

    output_file = args.output_file
    logging.info("***** Writing to output files *****")
    logging.info("Output File: %s", output_file)

    writer = FileWriter(output_file, args.num_splits)
    data_schema = {"input_ids": {"type": "int64", "shape": [-1]},
                   "attention_mask": {"type": "int64", "shape": [-1]},
                   "labels": {"type": "int64", "shape": [-1]}
                   }
    data_columns = args.data_columns
    need_del_keys = set(data_columns) - set(data_schema.keys())
    for need_del_key in need_del_keys:
        del data_schema[need_del_key]
    writer.add_schema(data_schema, "lm-schema")

    total_written = 0
    logging.info("***** Reading from  %s *****", input_file)
    text_total = ''
    with open(input_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            text_total = text_total + line
        sentence = text_total.strip().split("\t")
        block_size = args.max_length
        total_ids = tokenizer.encode(sentence[0])
        total_length = len(total_ids)
        total_length = (total_length // block_size) * block_size
        print("total_length", total_length)
        for i in range(total_length // block_size):
            ids = total_ids[block_size*i:block_size*(i+1)]

            output = create_instance(tokenizer, sentence, ids, args.max_length)

            write_instance_to_file(writer, instance=output)
            total_written += 1

    writer.commit()
    logging.info("Wrote %d total instances", total_written)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
