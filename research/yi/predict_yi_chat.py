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
# -*- coding: utf-8 -*-

"""Script to run chat on Yi-34B-Chat model."""
import argparse
import os
import numpy as np

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model

from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, LlamaTokenizer, LlamaForCausalLM
from mindformers import init_context
from mindformers.tools.utils import str2bool, get_real_rank
from mindformers.trainer.utils import get_last_checkpoint


def main(config):
    """main function."""
    # 多batch输入
    inputs = config.prompt.split()

    # set model config
    insertconfig = MindFormerConfig(config.yaml_file)

    # 初始化环境
    init_context(use_parallel=insertconfig.use_parallel,
                 context_config=insertconfig.context,
                 parallel_config=insertconfig.parallel)

    model_config = LlamaConfig(**insertconfig.model.model_config)
    model_config.parallel_config = TransformerOpParallelConfig(**insertconfig.parallel_config)
    model_config.batch_size = len(inputs)
    model_config.use_past = config.use_past
    model_config.seq_length = config.seq_length
    if config.checkpoint_path and not insertconfig.use_parallel:
        model_config.checkpoint_name_or_path = config.checkpoint_path # 如果本地已有ckpt，可加绝对路径：/path/to/model.ckpt

    # build tokenizer
    tokenizer = LlamaTokenizer(**insertconfig.processor.tokenizer)
    model = LlamaForCausalLM(model_config)

    # if use parallel, load distributed checkpoints
    if insertconfig.use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(config.checkpoint_path, "rank_{}".format(get_real_rank()))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # # shard model and load sharded ckpt
        warm_up_model = Model(model)
        input_ids = ms.Tensor(np.ones(shape=(model_config.batch_size, model_config.seq_length)), ms.int32)
        if model_config.use_past:
            infer_data = model.prepare_inputs_for_predict_layout(input_ids)
            warm_up_model.infer_predict_layout(*infer_data)
        else:
            warm_up_model.infer_predict_layout(input_ids)
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(model, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    with open('text_generation_result.txt', 'w') as writers:
        for prompt in inputs:
            message = [{"role": "user", "content": prompt}]
            input_id = tokenizer.apply_chat_template(conversation=message, tokenize=True, add_generation_prompt=True)
            outputs = model.generate(input_id,
                                     max_length=model_config.max_decode_length,
                                     do_sample=False)
            result = tokenizer.decode(outputs[0][len(input_id):], skip_special_tokens=True)
            print(result)
            writers.write(f'text_generation:\n{result}\n')
    writers.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='yi_34b', type=str,
                        help='which model to use.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--seq_length', default=16384, type=int,
                        help='predict max length')
    parser.add_argument('--prompt', default='', type=str,
                        help='input prompt')
    args = parser.parse_args()
    main(args)
