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
"""Qwen7B predict script"""

import argparse

from mindformers import Trainer
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.tools.utils import str2bool

# pylint: disable=W0611
import qwen_model
# pylint: disable=W0611
import qwen_tokenizer


def context_init(use_parallel=False, optimizer_parallel=False, device_id=0):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=device_id)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                enable_parallel_optimizer=optimizer_parallel,
                                                full_batch=True)
    init_context(use_parallel=use_parallel,
                 context_config=context_config,
                 parallel_config=parallel_config)


def main(task='text_generation',
         config='run_qwen_7b.yaml',
         run_mode='predict',
         pet_method='',
         use_parallel=False,
         ckpt=None,
         predict_data='',
         max_length=512,
         op=True,
         device_id=0):
    """main function."""
    # 环境初始化
    context_init(use_parallel, op, device_id)

    # 定义任务，预先准备好相应数据集
    if run_mode == 'predict':
        task = Trainer(args=config,
                       task=task,
                       pet_method=pet_method)
        # prompt = "<s><|User|>:{}<eoh>\n<|Bot|>:".format(predict_data)
        prompt = predict_data
        result = task.predict(input_data=prompt,
                              predict_checkpoint=ckpt, max_length=int(max_length), seq_length=max_length)
        print(result)
    else:
        raise NotImplementedError(f"run_mode '${run_mode}' not supported yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='text_generation', type=str,
                        help='set task type.')
    parser.add_argument('--config', default='run_qwen_7b.yaml', type=str,
                        help='set task type.')
    parser.add_argument('--run_mode', default='predict', type=str,
                        help='set run mode for model.')
    parser.add_argument('--pet_method', default='', type=str,
                        help='set pet method for low parameter finetune.')
    parser.add_argument('--load_checkpoint', default='', type=str,
                        help='checkpoint name or dir to load.')
    parser.add_argument('--predict_data', default='', type=str,
                        help='input predict data.')
    parser.add_argument('--predict_length', default=512, type=int,
                        help='max length for predict output.')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='open parallel for model.')
    parser.add_argument('--optimizer_parallel', default=False, type=str2bool,
                        help='whether use optimizer parallel. Default: False')
    parser.add_argument('--device_id', default=0, type=int,
                        help='ID of the target device, the value must be in [0, device_num_per_host-1]')
    args = parser.parse_args()

    main(task=args.task,
         config=args.config,
         run_mode=args.run_mode,
         pet_method=args.pet_method,
         use_parallel=args.use_parallel,
         ckpt=args.load_checkpoint,
         predict_data=args.predict_data,
         max_length=args.predict_length,
         op=args.optimizer_parallel,
         device_id=args.device_id)
