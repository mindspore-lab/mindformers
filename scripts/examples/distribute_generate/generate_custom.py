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
"""custom example of distribute generate"""
import argparse
import os

import numpy as np
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model

from mindformers import (AutoConfig, AutoModel, AutoTokenizer, ContextConfig,
                         ParallelContextConfig, init_context)
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.tools.utils import str2bool
from mindformers.trainer.utils import get_last_checkpoint


def context_init(use_parallel=False, device_id=0):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=device_id)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(
            parallel_mode='SEMI_AUTO_PARALLEL',     # 默认使用半自动并行模式
            gradients_mean=False,                   # 推理不涉及梯度平均
            full_batch=True                         # 半自动并行默认开启full batch
        )
    # 初始化context环境
    rank_id, device_num = init_context(
        use_parallel=use_parallel,
        context_config=context_config,
        parallel_config=parallel_config
    )
    print(f"Context inited for rank {rank_id}; total device num is {device_num}.")


def load_sharding_checkpoint(checkpoint_path, network, model_config):
    """compile and shard model, load distribute checkpoints."""
    if not os.path.isdir(checkpoint_path):
        raise ValueError(f"checkpoint_path {checkpoint_path} is not a directory, which is required for distribute "
                         "generate, please check your input checkpoint path.")
    # find the sharded ckpt path for this rank
    ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(os.getenv("RANK_ID", "0")))
    ckpt_path = get_last_checkpoint(ckpt_path)
    print(f"ckpt path: {str(ckpt_path)}", flush=True)

    # shard model and load sharded ckpt
    model = Model(network)
    model.infer_predict_layout(ms.Tensor(np.ones(shape=(model_config.batch_size, model_config.seq_length)), ms.int32))
    checkpoint_dict = load_checkpoint(ckpt_path)
    not_load_network_params = load_param_into_net(network, checkpoint_dict)
    print(f"Network parameters are not loaded: {str(not_load_network_params)}", flush=True)


def build_input_ids(input_text, tokenizer, **kwargs):
    """build input ids."""
    # 按需重写此函数以实现prompt构建，多轮对话构建等内容
    input_ids = tokenizer(input_text, padding=True, **kwargs)['input_ids']
    return input_ids


def main(args):
    """main function."""
    # 1. 初始化context环境
    print("---------------Init Context---------------", flush=True)
    context_init(args.use_parallel, args.device_id)

    # 2. 自定义模型配置
    print("---------------Set Model Config---------------", flush=True)
    # 2.1 获取模型默认参数
    model_config = AutoConfig.from_pretrained(args.model_type)
    # 2.2 自定义修改模型推理所需参数
    if args.batch_size is not None:
        model_config.batch_size = args.batch_size   # 增量推理时kv past的batch大小
    if args.seq_length is not None:
        model_config.seq_length = args.seq_length   # 模型推理支持的最长seq length
    if args.use_past is not None:
        model_config.use_past = args.use_past       # 是否开启增量推理
    # 2.3 配置模型切分策略，当前暂不支持pipeline并行策略
    parallel_config = TransformerOpParallelConfig(
        data_parallel=args.data_parallel,
        model_parallel=args.model_parallel
    )
    model_config.parallel_config = parallel_config
    # 2.4 分布式推理时需通过分布式接口加载权重，移除原权重路径以避免在模型实例化时加载
    if args.use_parallel:
        model_config.checkpoint_name_or_path = None
    if args.checkpoint_path and not args.use_parallel:
        model_config.checkpoint_name_or_path = args.checkpoint_path
    print(f"model config is: {model_config}")

    # 3. 实例化模型与Tokenizer
    print("---------------Build Model & Tokenizer---------------", flush=True)
    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    # build model from config
    network = AutoModel.from_config(model_config)

    # 4. 分布式下，模型编译切分并加载权重
    # if use parallel, load distributed checkpoints
    if args.use_parallel:
        print("---------------Load Sharding Checkpoints---------------", flush=True)
        load_sharding_checkpoint(args.checkpoint_path, network, model_config)

    # 5. 完成准备工作，调用推理流程
    print("---------------Start Generate---------------", flush=True)
    # while True:
    input_text = args.input_data
    print(">>> input_text: ", input_text)
    input_ids = build_input_ids(input_text, tokenizer)
    print(">>> input_ids: ", input_ids)
    output = network.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(tokenizer.decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, type=str,
                        help='which model to use. supports model name or model dir path.')
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='whether use parallel.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--data_parallel', default=1, type=int,
                        help='data parallel num for distribute generate.')
    parser.add_argument('--model_parallel', default=1, type=int,
                        help='model parallel num for distribute generate.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size of generate.')
    parser.add_argument('--seq_length', default=None, type=int,
                        help='seq length of model.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--input_data', default=None, type=str,
                        help='input text for generate, multi input text will be batched.')
    parser.add_argument('--max_new_tokens', default=128, type=int,
                        help='max new tokens to generate.')
    parser.add_argument('--do_sample', default=True, type=str2bool,
                        help='whether do sample.')
    parser.add_argument('--top_k', default=3, type=int,
                        help='top_k for do sample.')
    parser.add_argument('--top_p', default=1.0, type=float,
                        help='top_p for do sample.')
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='temperature for do sample.')
    parser.add_argument('--repetition_penalty', default=1.0, type=float,
                        help='repetition penalty for generate post process.')
    args_ = parser.parse_args()

    if os.path.isfile(args_.input_data):
        # 解析文本文件内容，每行作为一条文本输入，组成batch
        input_list = []
        with open(args_.input_data, 'r') as fp:
            input_list = []
            for line in fp:
                line = line.strip('\n')
                line = line.replace(r"\n", "\n")
                input_list.append(line)
        args_.input_data = input_list

    main(args_)
