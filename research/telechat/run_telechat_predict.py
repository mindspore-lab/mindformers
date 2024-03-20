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
"""Telechat predict scripts."""
import argparse
import json
from transformers import AutoTokenizer
from telechat_config import TelechatConfig
from telechat import TelechatForCausalLM
from mindformers import MindFormerConfig, TransformerOpParallelConfig
from mindformers import init_context
from mindformers.tools.utils import str2bool

def chat():
    """main function."""
    inputs = []
    input_file = open(args.input_file, 'r', encoding='utf-8')
    for line in input_file.readlines():
        dic = json.loads(line)
        data = "<_user>"+dic["input"]+"<_bot>"
        inputs.append(data)
    input_file.close()
    # set model config
    config = MindFormerConfig(args.yaml_file)
    config.use_parallel = False
    # 初始化环境
    init_context(context_config=config.context)

    model_config = TelechatConfig(**config.model.model_config)
    model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    model_config.batch_size = 1
    model_config.use_past = args.use_past
    model_config.use_flash_attention = False

    if args.checkpoint_path and not config.use_parallel:
        model_config.checkpoint_name_or_path = args.checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_file_path, fast_tokenizer=True,
                                              trust_remote_code=True, padding_side="right")
    # build model from config
    model = TelechatForCausalLM(model_config)
    for input_data in inputs:
        inputs_ids = tokenizer(input_data, max_length=model_config.seq_length, padding="max_length")["input_ids"]
        output = model.generate(inputs_ids,
                                max_length=model_config.max_decode_length,
                                do_sample=model_config.do_sample,
                                top_k=model_config.top_k,
                                top_p=model_config.top_p,
                                max_new_tokens=200)
        print("answer:", tokenizer.decode(output[0], skip_special_tokens=True))
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='', type=str,
                        help='input to infer.')
    parser.add_argument('--vocab_file_path', default='', type=str,
                        help='which model to use.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    args = parser.parse_args()
    chat()
