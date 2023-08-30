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
"""lite infer main."""

import sys
import argparse
from threading import Thread

# Avoid bugs when mslite and mindspore are not built from same commit, which may cause running error.
# pylint: disable=W0611
import mindspore_lite as mslite

from mindformers.models.base_tokenizer import Tokenizer
from mindformers.models import BloomTokenizer, LlamaTokenizer
from mindformers.models import ChatGLMTokenizer, ChatGLM2Tokenizer
from mindformers.pipeline import pipeline
from mindformers.generation import TextIteratorStreamer
from mindformers.tools.utils import str2bool
from mindformers.inference import InferConfig, InferTask


def pipeline_from_model_paths(args_, tokenizer):
    """build infer pipeline for model paths."""
    lite_pipeline = pipeline(
        task="text_generation",
        model=(args_.prefill_model_path, args_.increment_model_path),
        tokenizer=tokenizer,
        backend="mslite",
        model_name=args_.model_name,
        ge_config_path=args_.config_path,
        device_id=args_.device_id,
        infer_seq_length=args_.seq_length,
    )
    return lite_pipeline


def pipeline_from_model_name(args_, tokenizer):
    """build infer pipeline for model name."""
    lite_pipeline = pipeline(
        task="text_generation",
        model=args_.model_name,
        tokenizer=tokenizer,
        backend="mslite",
        ge_config_path=args_.config_path,
        device_id=args_.device_id,
        infer_seq_length=args_.seq_length,
    )
    return lite_pipeline


def pipeline_from_model_dir(args_, tokenizer):
    """build infer pipeline for model dir."""
    lite_pipeline = pipeline(
        task="text_generation",
        model=args_.model_dir,
        tokenizer=tokenizer,
        backend="mslite",
        model_name=args_.model_name,
        ge_config_path=args_.config_path,
        device_id=args_.device_id,
        infer_seq_length=args_.seq_length,
    )
    return lite_pipeline


def pipeline_from_infer_config(args_, tokenizer):
    """build infer pipeline for infer config."""
    lite_config = InferConfig(
        prefill_model_path=args_.prefill_model_path,
        increment_model_path=args_.increment_model_path,
        model_type="mindir",
        model_name=args_.model_name,
        ge_config_path=args_.config_path,
        device_id=args_.device_id,
        infer_seq_length=args_.seq_length,
    )
    lite_pipeline = InferTask.get_infer_task("text_generation", lite_config, tokenizer=tokenizer)
    return lite_pipeline


# the model name list that mslite inference has supported.
LITE_SUPPORT_MODELS = ('bloom', 'glm', 'glm2', 'llama')


def get_tokenizer(model_name: str) -> Tokenizer:
    """get tokenizer with model name."""
    tokenizer = None
    if model_name == 'bloom':
        tokenizer = BloomTokenizer.from_pretrained("bloom_560m")
    elif model_name == 'glm':
        tokenizer = ChatGLMTokenizer.from_pretrained("glm_6b")
    elif model_name == 'glm2':
        tokenizer = ChatGLM2Tokenizer.from_pretrained("glm2_6b")
    elif model_name == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained("llama_7b")
    else:
        raise ValueError(
            f"model must be in {LITE_SUPPORT_MODELS} when getting tokenizer, but got input {model_name}.")
    return tokenizer


def infer_main(args_):
    """lite infer main."""
    tokenizer = get_tokenizer(args_.model_name.lower())
    lite_pipeline = pipeline_from_model_paths(
        args_, tokenizer
    )

    while True:
        user_input = input("Please enter your predict data: \n")
        if user_input == "exit":
            print("Task is over.")
            sys.exit()
        output = lite_pipeline(user_input, is_sample_acceleration=args_.is_sample_acceleration,
                               add_special_tokens=args_.add_special_tokens)
        print(output)


def infer_stream_main(args_):
    """main entry for infer stream."""
    tokenizer = get_tokenizer(args_.model_name.lower())
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    lite_pipeline = pipeline_from_model_paths(
        args_, tokenizer
    )

    while True:
        user_input = input("Please enter your predict data(input 'exit' to quit): \n")
        if user_input == "exit":
            print("Quit now, this may take a while.")
            sys.exit()
        generation_kwargs = dict(inputs=user_input,
                                 streamer=streamer,
                                 is_sample_acceleration=args_.is_sample_acceleration,
                                 add_special_tokens=args_.add_special_tokens)
        thread = Thread(target=lite_pipeline, kwargs=generation_kwargs)
        thread.start()
        output = ""
        for new_text in streamer:
            print(new_text)
            output += new_text
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device_id', default=0, type=int,
        help='ID of the target device, the value must be in [0, device_num_per_host-1], '
             'while device_num_per_host should be no more than 4096. Default: None')
    parser.add_argument(
        '--rank_id', default=0, type=int,
        help='ID of the target device, the value must be in [0, device_num_per_host-1], '
             'while device_num_per_host should be no more than 4096. Default: None')
    parser.add_argument(
        '--model_dir', default=None, type=str,
        help="This model dir path. "
             "Default: None")
    parser.add_argument(
        '--model_name', default="common", type=str,
        help=f"The model name, only supports name in {LITE_SUPPORT_MODELS}. "
             "Default: None")
    parser.add_argument(
        '--seq_length', default=2048, type=int,
        help="This model dir path. "
             "Default: None")
    parser.add_argument(
        '--prefill_model_path', default=None, type=str,
        help="This full model path. "
             "Default: None")
    parser.add_argument(
        '--increment_model_path', default=None, type=str,
        help="When use kv-cache, this is cache mode path. "
             "Default: None")
    parser.add_argument(
        '--config_path', default=None, type=str,
        help="ge config file path. "
             "Default: None")
    parser.add_argument(
        '--is_sample_acceleration', default=False, type=str2bool,
        help="Whether postprocess in graph or not."
             "Default: False")
    parser.add_argument(
        '--add_special_tokens', default=False, type=str2bool,
        help="Whether preprocess add special tokens or not."
             "Default: False")
    parser.add_argument(
        '--stream', default=False, type=str2bool,
        help="Whether decode in stream or not."
             "Default: False")

    args = parser.parse_args()
    if args.stream:
        infer_stream_main(args)
    else:
        infer_main(args)
