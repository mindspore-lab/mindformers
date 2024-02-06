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
"""Telechat Train/Finetune scripts."""
import os
import sys
import shutil
import argparse

# pylint: disable=W0611
from mindformers import Trainer, MindFormerConfig
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.tools.utils import check_in_modelarts, set_remote_save_url, str2bool, check_shared_disk
from mindformers.tools.logger import logger
from mindformers.tools.cloud_adapter import cloud_monitor
from mindformers.core.context import build_context
from mindformers.tools import get_output_root_path
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from telechat_config import TelechatConfig
from telechat import TelechatForCausalLM
MindFormerRegister.register_cls(TelechatConfig, MindFormerModuleType.CONFIG)
MindFormerRegister.register_cls(TelechatForCausalLM, MindFormerModuleType.MODELS)


if check_in_modelarts():
    import moxing as mox

sys.path.insert(0, os.getcwd().split('research')[0])

def clear_auto_trans_output(config):
    """clear transformed_checkpoint and strategy"""
    if check_in_modelarts():
        obs_strategy_dir = os.path.join(config.remote_save_url, "strategy")
        if mox.file.exists(obs_strategy_dir) and config.local_rank == 0:
            mox.file.remove(obs_strategy_dir, recursive=True)
            mox.file.make_dirs(obs_strategy_dir)
        obs_transformed_ckpt_dir = os.path.join(config.remote_save_url, "transformed_checkpoint")
        if mox.file.exists(obs_transformed_ckpt_dir) and config.local_rank == 0:
            mox.file.remove(obs_transformed_ckpt_dir, recursive=True)
            mox.file.make_dirs(obs_transformed_ckpt_dir)
    else:
        strategy_dir = os.path.join(get_output_root_path(), "strategy")
        if os.path.exists(strategy_dir) and config.local_rank == 0:
            shutil.rmtree(strategy_dir)
            os.makedirs(strategy_dir, exist_ok=True)
        transformed_ckpt_dir = os.path.join(get_output_root_path(), "transformed_checkpoint")
        if os.path.exists(transformed_ckpt_dir) and config.local_rank == 0:
            shutil.rmtree(transformed_ckpt_dir)
            os.makedirs(transformed_ckpt_dir, exist_ok=True)


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


@cloud_monitor()
def main(task='text_generation',
         config='run_telechat_7b_finetune_910b.yaml',
         run_mode='train',
         seq_length=None,
         mode=None,
         use_parallel=None,
         device_id=None,
         ckpt=None,
         strategy=None,
         auto_trans_ckpt=None,
         resume=False,
         train_dataset='',
         remote_save_url=None,
         vocab_file=None):
    """main function."""

    assert os.path.exists(config) and config.endswith(('.yaml', '.yml'))

    # init config
    config = MindFormerConfig(os.path.realpath(config))
    if seq_length is not None:
        config.model.model_config.seq_length = seq_length
    if mode is not None:
        config.context.mode = mode
        if mode:
            config.recompute_config.recompute = False
    if use_parallel is not None:
        config.use_parallel = use_parallel
    if device_id is not None:
        config.context.device_id = device_id
    if ckpt is None:
        ckpt = config.load_checkpoint
    if strategy is not None and os.path.exists(strategy):
        config.src_strategy_path_or_dir = strategy
    if auto_trans_ckpt is not None:
        config.auto_trans_ckpt = auto_trans_ckpt
    if vocab_file is not None:
        config.processor.tokenizer.vocab_file = vocab_file

    if config.output_dir != './output':
        raise ValueError("output_dir must be set to './output' and cannot be customized.")

    # init context
    build_context(config)

    if check_in_modelarts() and remote_save_url:
        logger.info("remote_save_url is %s, the output file will be uploaded to here.", remote_save_url)
        set_remote_save_url(remote_save_url)
        config.remote_save_url = remote_save_url

    if run_mode in ['train', 'finetune']:
        config.model.model_config.use_past = False

    if config.auto_trans_ckpt:
        if config.device_num <= 8 or check_shared_disk(config.output_dir) or check_in_modelarts():
            clear_auto_trans_output(config)
        else:
            raise ValueError("When device num > 8 and auto_trans_ckpt is set to True,"
                             "the output_dir should be a shared directory that can be accessed by all nodes."
                             f"but {os.path.abspath(config.output_dir)} is not a shared directory.")

    # start task
    if run_mode == 'train':
        trainer = Trainer(args=config,
                          task=task,
                          train_dataset=train_dataset)
        trainer.train(train_checkpoint=ckpt, auto_trans_ckpt=config.auto_trans_ckpt, resume_training=resume)
    elif run_mode == 'finetune':
        trainer = Trainer(args=config,
                          task=task,
                          train_dataset=train_dataset)
        trainer.finetune(finetune_checkpoint=ckpt, auto_trans_ckpt=config.auto_trans_ckpt, resume_training=resume)
    else:
        raise ValueError("run_mode only support train and finetune.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='text_generation', type=str,
                        help='set task type.')
    parser.add_argument('--config', default='telechat/run_telechat_7b_910b_finetune.yaml', type=str,
                        help='set task type.')
    parser.add_argument('--run_mode', default='train', type=str,
                        help='set run mode for model.')
    parser.add_argument('--seq_length', default=None, type=int,
                        help='seq_length')
    parser.add_argument('--use_parallel', default=True, type=str2bool,
                        help='open parallel for model.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='device id set when run on single card. Default: 0')
    parser.add_argument('--mode', default=0, type=int,
                        help='0--Graph Mode; 1--Pynative Mode')
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help='checkpoint name or dir to load.')
    parser.add_argument('--src_strategy', default=None, type=str,
                        help='strategy of load_checkpoint')
    parser.add_argument('--auto_trans_ckpt', default=None, type=str2bool,
                        help='whether to transform checkpoint to the checkpoint matching current distribute strategy.')
    parser.add_argument('--resume', default=None, type=str2bool,
                        help='whether resume training.')
    parser.add_argument('--train_dataset', default='', type=str,
                        help='set train dataset.')
    parser.add_argument('--remote_save_url', default='', type=str,
                        help='whether use optimizer parallel. Default: None')
    parser.add_argument('--vocab_file', default=None, type=str,
                        help='tokenizer model')
    args = parser.parse_args()

    main(task=args.task,
         config=args.config,
         run_mode=args.run_mode,
         seq_length=args.seq_length,
         mode=args.mode,
         use_parallel=args.use_parallel,
         device_id=args.device_id,
         ckpt=args.load_checkpoint,
         strategy=args.src_strategy,
         auto_trans_ckpt=args.auto_trans_ckpt,
         resume=args.resume,
         train_dataset=args.train_dataset,
         remote_save_url=args.remote_save_url,
         vocab_file=args.vocab_file)
