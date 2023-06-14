# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Run MindFormer."""
import argparse
import os
from pprint import pprint

import numpy as np

import mindspore as ms
from mindspore.common import set_seed

from mindformers.tools.register import MindFormerConfig, ActionDict
from mindformers.core.parallel_config import build_parallel_config
from mindformers.tools.utils import str2bool, set_remote_save_url, check_in_modelarts
from mindformers.core.context import build_context, build_profile_cb
from mindformers.trainer import build_trainer
from mindformers.tools.cloud_adapter import cloud_monitor
from mindformers.tools.logger import logger
from mindformers.mindformer_book import MindFormerBook


SUPPORT_MODEL_NAMES = MindFormerBook().get_model_name_support_list()


def update_checkpoint_config(config, is_train=True):
    """update checkpoint config depending on is_train"""
    if (is_train and config.resume_training) or config.auto_trans_ckpt or os.path.isdir(config.load_checkpoint):
        logger.info("Leave load_checkpoint may because: ")
        logger.info("1. resume training need resume training info. ")
        logger.info("2. need load distributed shard checkpoint. ")
        if not config.load_checkpoint:
            config.load_checkpoint = config.model.model_config.checkpoint_name_or_path
        config.model.model_config.checkpoint_name_or_path = None
    else:
        if config.run_mode == 'train':
            config.model.model_config.checkpoint_name_or_path = None
        elif config.run_mode == 'finetune':
            config.model.model_config.checkpoint_name_or_path = config.load_checkpoint
        config.load_checkpoint = None


@cloud_monitor()
def main(config):
    """main."""
    # init context
    build_context(config)

    if ms.context.get_auto_parallel_context("parallel_mode") not in ["semi_auto_parallel", "auto_parallel"]:
        set_seed(config.seed)
        np.random.seed(config.seed)

    # build context config
    logger.info(".........Build context config..........")
    build_parallel_config(config)
    logger.info("context config is: %s", config.parallel_config)
    logger.info("moe config is: %s", config.moe_config)

    if config.run_mode == 'train':
        update_checkpoint_config(config)

    if config.run_mode == 'finetune':
        if not config.load_checkpoint:
            raise ValueError("if run status is finetune, "
                             "load_checkpoint must be input")
        update_checkpoint_config(config)

    if config.run_mode == 'eval':
        update_checkpoint_config(config, is_train=False)

    if config.run_mode == "predict" and config.load_checkpoint:
        if config.auto_trans_ckpt:
            logger.warning("In the predict mode, distributed loading weights are not supported."
                           "Only single-card reasoning is supported")
        config.model.model_config.checkpoint_name_or_path = config.load_checkpoint
        config.load_checkpoint = None

    # remote save url
    if check_in_modelarts() and config.remote_save_url:
        logger.info("remote_save_url is %s, the output file will be uploaded to here.", config.remote_save_url)
        set_remote_save_url(config.remote_save_url)

    # define callback and add profile callback
    if config.profile:
        config.profile_cb = build_profile_cb(config)

    if config.local_rank % 8 == 0:
        pprint(config)

    trainer = build_trainer(config.trainer)
    if config.run_mode == 'train' or config.run_mode == 'finetune':
        trainer.train(config, is_full_config=True)
    elif config.run_mode == 'eval':
        trainer.evaluate(config, is_full_config=True)
    elif config.run_mode == 'predict':
        trainer.predict(config, is_full_config=True)


if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default="configs/mae/run_mae_vit_base_p16_224_800ep.yaml",
        required=True,
        help='YAML config files')
    parser.add_argument(
        '--mode', default=None, type=int,
        help='Running in GRAPH_MODE(0) or PYNATIVE_MODE(1). Default: GRAPH_MODE(0).'
             'GRAPH_MODE or PYNATIVE_MODE can be set by `mode` attribute and both modes support all backends,'
             'Default: None')
    parser.add_argument(
        '--device_id', default=None, type=int,
        help='ID of the target device, the value must be in [0, device_num_per_host-1], '
             'while device_num_per_host should be no more than 4096. Default: None')
    parser.add_argument(
        '--device_target', default=None, type=str,
        help='The target device to run, support "Ascend", "GPU", and "CPU".'
             'If device target is not set, the version of MindSpore package is used.'
             'Default: None')
    parser.add_argument(
        '--run_mode', default=None, type=str,
        help='task running status, it support [train, finetune, eval, predict].'
             'Default: None')
    parser.add_argument(
        '--train_dataset_dir', default=None, type=str,
        help='dataset directory of data loader to train/finetune. '
             'Default: None')
    parser.add_argument(
        '--eval_dataset_dir', default=None, type=str,
        help='dataset directory of data loader to eval. '
             'Default: None')
    parser.add_argument(
        '--predict_data', default=None, type=str,
        help='input data for predict, it support real data path or data directory.'
             'Default: None')
    parser.add_argument(
        '--load_checkpoint', default=None, type=str,
        help="load model checkpoint to train/finetune/eval/predict, "
             "it is also support input model name, such as 'mae_vit_base_p16', "
             "please refer to https://gitee.com/mindspore/mindformers#%E4%BB%8B%E7%BB%8D."
             "Default: None")
    parser.add_argument(
        '--resume_training', default=None, type=str2bool,
        help="whether to load training context info, such as optimizer and epoch num")
    parser.add_argument(
        '--strategy_load_checkpoint', default=None, type=str,
        help='path to parallel strategy checkpoint to load, it support real data path or data directory.'
             'Default: None')
    parser.add_argument(
        '--remote_save_url', default=None, type=str,
        help='remote save url, where all the output files will tansferred and stroed in here. '
             'Default: None')
    parser.add_argument(
        '--seed', default=None, type=int,
        help='global random seed to train/finetune.'
             'Default: None')
    parser.add_argument(
        '--use_parallel', default=None, type=str2bool,
        help='whether use parallel mode. Default: None')
    parser.add_argument(
        '--profile', default=None, type=str2bool,
        help='whether use profile analysis. Default: None')
    parser.add_argument(
        '--options',
        nargs='+',
        action=ActionDict,
        help='override some settings in the used config, the key-value pair'
             'in xxx=yyy format will be merged into config file')
    parser.add_argument(
        '--epochs', default=None, type=int,
        help='train epochs.'
             'Default: None')
    parser.add_argument(
        '--batch_size', default=None, type=int,
        help='batch_size of datasets.'
             'Default: None')
    parser.add_argument(
        '--sink_mode', default=None, type=str2bool,
        help='whether use sink mode. '
             'Default: None')
    parser.add_argument(
        '--num_samples', default=None, type=int,
        help='number of datasets samples used.'
             'Default: None')

    args_ = parser.parse_args()
    if args_.config is not None:
        args_.config = os.path.join(work_path, args_.config)
    config_ = MindFormerConfig(args_.config)
    if args_.device_id is not None:
        config_.context.device_id = args_.device_id
    if args_.device_target is not None:
        config_.context.device_target = args_.device_target
    if args_.mode is not None:
        config_.context.mode = args_.mode
    if args_.run_mode is not None:
        config_.run_mode = args_.run_mode
    if args_.seed is not None:
        config_.seed = args_.seed
    if args_.use_parallel is not None:
        config_.use_parallel = args_.use_parallel
    if args_.load_checkpoint is not None:
        config_.load_checkpoint = args_.load_checkpoint
    if args_.resume_training is not None:
        config_.resume_training = args_.resume_training
    if args_.strategy_load_checkpoint is not None:
        if os.path.isdir(args_.strategy_load_checkpoint):
            ckpt_list = [os.path.join(args_.strategy_load_checkpoint, file)
                         for file in os.listdir(args_.strategy_load_checkpoint) if file.endwith(".ckpt")]
            args_.strategy_load_checkpoint = ckpt_list[0]
        config_.parallel.strategy_ckpt_load_file = args_.strategy_load_checkpoint
    if args_.remote_save_url is not None:
        config_.remote_save_url = args_.remote_save_url
    if args_.profile is not None:
        config_.profile = args_.profile
    if args_.options is not None:
        config_.merge_from_dict(args_.options)
    assert config_.run_mode in ['train', 'eval', 'predict', 'finetune'], \
        f"run status must be in {['train', 'eval', 'predict', 'finetune']}, but get {config_.run_mode}"
    if args_.train_dataset_dir:
        config_.train_dataset.data_loader.dataset_dir = args_.train_dataset_dir
    if args_.eval_dataset_dir:
        config_.eval_dataset.data_loader.dataset_dir = args_.eval_dataset_dir
    if config_.run_mode == 'predict':
        if args_.predict_data is None:
            logger.info("dataset by config is used as input_data.")
        elif os.path.isdir(args_.predict_data) and os.path.exists(args_.predict_data):
            predict_data = [os.path.join(root, file)
                            for root, _, file_list in os.walk(os.path.join(args_.predict_data)) for file in file_list
                            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")
                            or file.endswith(".JPEG") or file.endswith("bmp")]
            args_.predict_data = predict_data
        config_.input_data = args_.predict_data
    if args_.epochs is not None:
        config_.runner_config.epochs = args_.epochs
    if args_.batch_size is not None:
        config_.runner_config.batch_size = args_.batch_size
    if args_.sink_mode is not None:
        config_.runner_config.sink_mode = args_.sink_mode
    if args_.num_samples is not None:
        if config_.train_dataset and config_.train_dataset.data_loader:
            config_.train_dataset.data_loader.num_samples = args_.num_samples
        if config_.eval_dataset and config_.eval_dataset.data_loader:
            config_.eval_dataset.data_loader.num_samples = args_.num_samples
    main(config_)
