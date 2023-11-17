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
"""test wizardcoder mslite"""
import argparse
from mindspore import context
from mindformers.pipeline import pipeline
from wizardcoder_tokenizer import WizardCoderTokenizer


def main(args):
    context.set_context(device_id=args.device_id, mode=0, device_target="Ascend")
    tokenizer = WizardCoderTokenizer(
        vocab_file=args.tokenizer_path + "vocab.json",
        merge_file=args.tokenizer_path + "merges.txt"
    )
    model_path = (f"wizardcoder-15b_mslite_inc/prefill_seq{args.seq_length}_bs{args.batch_size}_graph.mindir",
                  f"wizardcoder-15b_mslite_inc/decode_seq{args.seq_length}_bs{args.batch_size}_graph.mindir")
    ge_config_path = "lite.ini"
    pipeline_task = pipeline(task="text_generation", model=model_path, backend="mslite", tokenizer=tokenizer,
                             ge_config_path=ge_config_path, model_type="mindir", infer_seq_length=args.seq_length,
                             add_special_tokens=False, device_id=args.device_id)

    input_data_list = [["使用python编写快速排序代码"] * args.batch_size] * 2

    for input_data in input_data_list:
        outputs, generate_length, _, elapsed_time_no_tokenizer = \
            pipeline_task(input_data,
                          do_sample=False,
                          max_length=2048,
                          eos_token_id=0,
                          pad_token_id=49152,
                          skip_special_tokens=True)
        print(outputs[0])
        print(f"generate length:{generate_length}, "
              f"time:{elapsed_time_no_tokenizer}, "
              f"speed: {generate_length / elapsed_time_no_tokenizer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch_size')
    parser.add_argument('--seq_length', default=2048, type=int,
                        help='batch_size')
    parser.add_argument('--tokenizer_path', default='', type=str,
                        help='tokenizer_path')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    opt = parser.parse_args()
    main(opt)
