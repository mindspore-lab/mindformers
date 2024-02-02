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
"""lite infer main."""
import os
import sys
import time
import argparse
from threading import Thread

# pylint: disable=C0413
# pylint: disable=W0611
# ms and lite using the same function in some modules
# import mindspore_lite first to using lite rather than ms module
# mindspore_lite and mindspore should from the same commit
import mindspore_lite
import iflytekspark_config
from mindformers.tools.utils import str2bool
from mindformers.auto_class import AutoConfig
from mindformers.inference import InferConfig
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# import local class
from iflytekspark_streamer import IFlytekSparkStreamer
from iflytekspark_generator_infer import IFlytekSparkGeneratorInfer
from iflytekspark_tokenizer import IFlytekSparkTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pipeline_from_infer_config(args_, tokenizer):
    """build infer pipeline for infer config."""
    prefill_model_path = args_.prefill_model_path.format(rank)
    increment_model_path = args_.decode_model_path.format(rank)
    sample_model_path = args_.sample_model_path.format(rank)
    lite_config = InferConfig(
        prefill_model_path=prefill_model_path,
        increment_model_path=increment_model_path,
        model_type="mindir",
        ge_config_path=(args_.prefill_model_config_path, args_.decode_model_config_path),
        device_id=args_.device_id,
        rank_id=args_.rank_id,
        infer_seq_length=args_.max_seq_len,
        is_dynamic=is_dynamic,
        sample_model_path=sample_model_path,
        sample_config_path=args_.sample_model_config_path,
    )
    lite_pipeline = IFlytekSparkGeneratorInfer(lite_config, tokenizer)
    return lite_pipeline


def infer_main(args_):
    """inference main function"""
    print("mindspore_lite version:", mindspore_lite.__version__, flush=True)
    print("initialize... ", flush=True)
    # init tokenizer
    tokenizer = IFlytekSparkTokenizer(args_.tokenizer_file)

    # init pipeline and streamer
    lite_pipeline = pipeline_from_infer_config(args_, tokenizer)
    if args_.stream_return:
        streamer = IFlytekSparkStreamer(tokenizer, skip_prompt=True)

    # warm up to compile models
    print("warm up... ", flush=True)
    generation_kwargs = dict(inputs=["初始化"] * args_.batch_size,
                             do_sample=do_sample,
                             top_k=top_k,
                             top_p=top_p,
                             temperature=temperature,
                             repetition_penalty=repetition_penalty,
                             repetition_penalty_increase=repetition_penalty_increase,
                             eos_token_id=eos_token_id,
                             pad_token_id=pad_token_id,
                             max_length=16,
                             add_special_tokens=False,
                             kv_cache_size=kv_cache_size)
    _ = lite_pipeline(**generation_kwargs)

    # catch user inputs string from cmd line
    generation_kwargs["max_length"] = args_.max_out_lenth
    if args_.stream_return:
        generation_kwargs["streamer"] = streamer

    print("generate... ", flush=True)
    input_list = []
    # support json or text file, take it as text file by default.
    if args_.input_file.endswith('json'):
        import jsonlines
        with jsonlines.open(args_.input_file) as f:
            for line in f:
                input_list.append(args_.prompt.format(line["input"]))
    else:
        with open(args_.input_file) as f:
            for line in f.readlines():
                user_input = line[:-1]
                input_list.append(args_.prompt.format(user_input))
    start_gen = time.time()
    for input_line in input_list:
        question_list = []
        start_one_line = time.time()
        # 输入准备，模拟多batch
        for i in range(args_.batch_size):
            user_input = input_line
            question_list.append(user_input)
        if user_input == "exit":
            print("Task is over.")
            exit()
        elif user_input == "":
            print("empty str, please retyping")
            continue
        generation_kwargs["inputs"] = question_list

        # 自回归生成
        if not args_.stream_return:
            output = lite_pipeline(**generation_kwargs)
            print(output)
        else:
            thread = Thread(target=lite_pipeline, kwargs=generation_kwargs)
            thread.start()
            output = [""] * args_.batch_size
            for new_text in streamer:
                print("streamer:", new_text)
                for i in range(len(new_text)):
                    output[i] += new_text[i]
            thread.join()
        end_one_line = time.time()
        print(f"one line infer time is: { (end_one_line-start_one_line)*1000 } ms")

    end_gen = time.time()
    print(f"total generate time is: { (end_gen-start_gen)*1000 } ms")
    print("=========== generate finish =================", flush=True)


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
        '--batch_size', default=1, type=int,
        help="the max output length of model"
             "Default: 128")
    parser.add_argument(
        '--max_seq_len', default=2048, type=int,
        help="the max seq length of exported model"
             "Default: None")
    parser.add_argument(
        '--prefill_model_path', default=None, type=str,
        help="This full model path. "
             "Default: None")
    parser.add_argument(
        '--decode_model_path', default=None, type=str,
        help="When use kv-cache, this is cache mode path. "
             "Default: None")
    parser.add_argument(
        '--prefill_model_config_path', default=None, type=str,
        help="ge config file path. needed by mslite backend. "
             "if run dynamic sequence length, it should be different with decode_model_config_path"
             "Default: None")
    parser.add_argument(
        '--decode_model_config_path', default=None, type=str,
        help="ge config file path. needed by mslite backend. "
             "if run dynamic sequence length, it should be different with prefill_model_config_path"
             "Default: None")
    parser.add_argument(
        '--sample_model_path', default=None, type=str,
        help="model path for sample logits. Default: None")
    parser.add_argument(
        '--sample_model_config_path', default=None, type=str,
        help="ge config file path for sample_model. Default: None")
    parser.add_argument(
        '--postprocess_config_path', default=None, type=str,
        help="Default: None")
    parser.add_argument(
        '--stream_return', default=False, type=str2bool,
        help="Whether enable token return as a stream, rather than return when all tokens are generated"
             "Default: False")
    parser.add_argument(
        '--max_out_lenth', default=128, type=int,
        help="the max output length of model"
             "Default: 128")
    parser.add_argument(
        '--tokenizer_file', default=None, type=str,
        help="user custom tokenizer file path"
             "Default: None")
    parser.add_argument(
        '--prompt', default="", type=str,
        help="promote add before user input")
    parser.add_argument(
        '--start_device_id', default=0, type=int,
        help="distribution run lite start device_id")
    parser.add_argument(
        '--input_file', default="", type=str,
        help="input_file for inference")
    args = parser.parse_args()
    args.rank_id = rank
    if comm.size > 1:
        args.device_id = args.start_device_id + rank

    postprocess_config = AutoConfig.from_pretrained(args.postprocess_config_path)
    pad_token_id = postprocess_config.pad_token_id
    eos_token_id = postprocess_config.eos_token_id
    top_k = postprocess_config.top_k
    top_p = postprocess_config.top_p
    repetition_penalty_increase = postprocess_config.repetition_penalty_increase
    temperature = postprocess_config.temperature
    repetition_penalty = postprocess_config.repetition_penalty
    do_sample = postprocess_config.do_sample
    kv_cache_size = postprocess_config.sparse_local_size
    is_dynamic = postprocess_config.is_dynamic

    print("prefill_model_path ", args.prefill_model_path)
    print("prefill_model_config_path ", args.prefill_model_config_path)
    print("decode_model_path ", args.decode_model_path)
    print("decode_model_config_path ", args.decode_model_config_path)
    print("sample_model_path ", args.sample_model_path)
    print("sample_model_config_path ", args.sample_model_config_path)
    print("stream_return ", args.stream_return)
    print("is_dynamic ", is_dynamic)
    print("max_out_lenth ", args.max_out_lenth)
    print("tokenizer_file ", args.tokenizer_file)
    print("batch_size ", args.batch_size)
    print("top_k ", top_k)
    print("top_p ", top_p)
    print("repetition_penalty ", repetition_penalty)
    print("repetition_penalty_increase ", repetition_penalty_increase)
    print("temperature ", temperature)
    print("do_sample ", do_sample)
    print("kv_cache_size ", kv_cache_size)

    infer_main(args)
