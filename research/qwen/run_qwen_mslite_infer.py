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
"""Example script to run Qwen exported model for MindSpore Lite."""
import os

# pylint: disable=W0611
import mindspore_lite
import mindspore as ms

from mindformers.pipeline import pipeline
from mindformers.inference import InferConfig, InferTask
from mindformers.tools.utils import str2bool
from qwen_tokenizer import QwenTokenizer


def get_mindir_path(export_path='output', full=True):
    """Return relative path to MINDIR file"""
    assert os.path.isdir(export_path)
    rank_id = os.getenv('RANK_ID', '0')

    mindir_path = "%s/mindir_%s_checkpoint/rank_%s_graph.mindir" % \
                (export_path, "full" if full else "inc", rank_id)
    assert os.path.isfile(mindir_path)
    var_path = "%s/mindir_%s_checkpoint/rank_%s_variables" % \
                (export_path, "full" if full else "inc", rank_id)
    assert os.path.isdir(var_path)
    return mindir_path


def create_mslite_pipeline(args):
    """Create MS lite inference pipeline."""
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    tokenizer = QwenTokenizer(pad_token='<|endoftext|>',
                              vocab_file='./run/qwen.tiktoken')

    prefill_model_path = get_mindir_path(args.mindir_root_dir, full=True)
    inc_model_path = get_mindir_path(args.mindir_root_dir, full=False)

    if args.device_id == -1:
        args.device_id = int(os.getenv('DEVICE_ID', '0'))

    rank_id = int(os.getenv('RANK_ID', '0'))

    print("Creating pipeline from (%s, %s)..." % (prefill_model_path, inc_model_path))
    lite_config = InferConfig(
        prefill_model_path=prefill_model_path,
        increment_model_path=inc_model_path,
        model_type="mindir",
        model_name="qwen",
        ge_config_path=args.ge_config_path,
        device_id=args.device_id,
        rank_id=rank_id,
        infer_seq_length=args.seq_length,
        paged_attention=args.paged_attention,
        pa_block_size=args.pa_block_size,
        pa_num_blocks=args.pa_num_blocks,
    )
    pipeline_task = InferTask.get_infer_task("text_generation", lite_config, tokenizer=tokenizer)
    return pipeline_task


def expand_input_list(input_list, batch_size):
    """Expand 'input_list' to a list of size 'batch_size'."""
    if len(input_list) < batch_size:
        repeat_time = batch_size // len(input_list) + 1
        input_list = input_list * repeat_time
    input_list = input_list[:batch_size]
    return input_list


def run_mslite_infer(pipeline_task, prompt, args):
    """Run MS lite inference with PROMPT and ARGS."""
    input_list = prompt
    if not isinstance(prompt, list):
        input_list = [prompt,]
    input_list = expand_input_list(input_list, args.batch_size)

    return pipeline_task.infer(
        input_list,
        max_length=args.predict_length,
        do_sample=args.do_sample,
        top_k=3,
        top_p=0.85,
        repetition_penalty=1.0,
        temperature=1.0,
        is_sample_acceleration=False,
        add_special_tokens=False,
        eos_token_id=151643,
        pad_token_id=151643)


def main(args):
    """Main function."""
    pipeline_task = create_mslite_pipeline(args)

    # to warm up the model
    run_mslite_infer(pipeline_task, "hello", args)

    outputs = run_mslite_infer(pipeline_task, args.predict_data, args)
    for output in outputs:
        print(output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default=-1, type=int,
                        help='ID of the target device, the value must be in [0, device_num_per_host-1]')
    parser.add_argument('--ge_config_path', default='./lite.ini', type=str,
                        help='ge config file path.')
    parser.add_argument('--mindir_root_dir', default='output', type=str,
                        help='root path of exported MINDIR models. Default: "output".')
    parser.add_argument('--seq_length', default=None, type=int, required=True,
                        help='seq_length')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch_size')
    parser.add_argument('--do_sample', default=None, type=str2bool,
                        help='do_sample')
    parser.add_argument('--predict_data', default='', type=str, required=True,
                        help='input predict data.')
    parser.add_argument('--predict_length', default=512, type=int,
                        help='max length for predict output.')
    parser.add_argument('--paged_attention', default=False, type=str2bool,
                        help="Whether use paged attention."
                        "Default: False")
    parser.add_argument('--pa_block_size', default=16, type=int,
                        help="Block size of paged attention."
                        "Default: 16")
    parser.add_argument('--pa_num_blocks', default=512, type=int,
                        help="The number of blocks of paged attention."
                        "Default: 512")
    args_ = parser.parse_args()

    main(args_)
