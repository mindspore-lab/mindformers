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
"""glm3-32k Train/Finetune/Eval/Predict scripts."""
import os
import argparse

from mindformers import Trainer, MindFormerConfig
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.tools.utils import str2bool
from mindformers.core.context import build_context

# pylint: disable=W0611
import glm32k
import glm32k_modules
from glm32k_tokenizer import ChatGLM32kTokenizer


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


def generate_prompt(instruction):
    """the prompt used for glm3-32k, which is corresponding to the training process"""
    return f"""<|user|>\n {instruction}<|assistant|>"""


def main(task='text_generation',
         config='run_glm32k.yaml',
         run_mode='train',
         pet_method='',
         use_parallel=False,
         ckpt=None,
         resume=False,
         train_dataset='',
         eval_dataset='',
         predict_data='',
         max_length=512,
         vocab_file=None,
         op=True,
         device_id=0):
    """main function."""

    assert os.path.exists(config) and config.endswith(('.yaml', '.yml'))

    # init config
    config = MindFormerConfig(os.path.realpath(config))
    if use_parallel is not None:
        config.use_parallel = use_parallel
    if vocab_file is not None:
        config.processor.tokenizer.vocab_file = vocab_file
    if device_id is not None:
        config.context.device_id = device_id
    config.parallel.enable_parallel_optimizer = op
    # init context
    build_context(config)

    # 定义任务，预先准备好相应数据集
    if run_mode == 'train':
        task = Trainer(args=config,
                       task=task,
                       train_dataset=train_dataset,
                       pet_method=pet_method)
        task.train(train_checkpoint=config.load_checkpoint,
                   auto_trans_ckpt=config.auto_trans_ckpt,
                   resume_training=resume)

    elif run_mode == 'finetune':
        task = Trainer(args=config,
                       task=task,
                       train_dataset=train_dataset,
                       pet_method=pet_method)
        task.finetune(finetune_checkpoint=config.load_checkpoint,
                      auto_trans_ckpt=config.auto_trans_ckpt,
                      resume_training=resume)

    elif run_mode == 'eval':
        task = Trainer(args=config,
                       task=task,
                       eval_dataset=eval_dataset,
                       pet_method=pet_method)
        task.evaluate(eval_checkpoint=config.load_checkpoint)

    elif run_mode == 'predict':
        task = Trainer(args=config,
                       task=task)
        prompt = generate_prompt(predict_data)
        result = task.predict(input_data=prompt,
                              predict_checkpoint=ckpt, max_length=int(max_length))
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='text_generation', type=str,
                        help='set task type.')
    parser.add_argument('--config', default='run_glm32k.yaml', type=str,
                        help='set task type.')
    parser.add_argument('--run_mode', default='train', type=str,
                        help='set run mode for model.')
    parser.add_argument('--pet_method', default='', type=str,
                        help='set pet method for low parameter finetune.')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='open parallel for model.')
    parser.add_argument('--load_checkpoint', default='', type=str,
                        help='load_checkpoint')
    parser.add_argument('--resume', default=False, type=str2bool,
                        help='whether resume training.')
    parser.add_argument('--train_dataset', default='', type=str,
                        help='set train dataset.')
    parser.add_argument('--eval_dataset', default='', type=str,
                        help='set eval dataset.')
    parser.add_argument('--predict_data', default='', type=str,
                        help='input predict data.')
    parser.add_argument('--predict_length', default=512, type=int,
                        help='max length for predict output.')
    parser.add_argument('--optimizer_parallel', default=True, type=str2bool,
                        help='whether use optimizer parallel. Default: False')
    parser.add_argument('--device_id', default=None, type=int,
                        help='ID of the target device, the value must be in [0, device_num_per_host-1]')
    parser.add_argument('--vocab_file', default=None, type=str,
                        help='tokenizer model')
    args = parser.parse_args()

    main(task=args.task,
         config=args.config,
         run_mode=args.run_mode,
         pet_method=args.pet_method,
         use_parallel=args.use_parallel,
         ckpt=args.load_checkpoint,
         resume=args.resume,
         train_dataset=args.train_dataset,
         eval_dataset=args.eval_dataset,
         predict_data=args.predict_data,
         max_length=args.predict_length,
         op=args.optimizer_parallel,
         device_id=args.device_id,
         vocab_file=args.vocab_file)
