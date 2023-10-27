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
"""Trainer Utils."""
import os
import sys
import time
import random
from glob import glob
from enum import Enum

import numpy as np

import mindspore as ms
from mindspore.communication.management import get_rank, get_group_size
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore import set_seed as ms_set_seed

from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.utils import check_in_modelarts, get_output_root_path, replace_tk_to_mindpet
from mindformers.tools.transform_ckpt import get_strategy
from mindformers.tools.cloud_adapter import mox_adapter

if check_in_modelarts():
    import moxing as mox

class BaseEnum(str, Enum):
    """
    Base Enum for MindFormers.
    """

    @classmethod
    def _missing_(cls, value):
        """Enum with more explicit error message for missing values."""
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class SaveIntervalStrategy(BaseEnum):
    """
    Stores the acceptable string identifiers for save checkpoint monitor.
    """
    NO = "no"
    STEPS = "steps"
    SECONDS = "seconds"


class LRType(BaseEnum):
    """
    Stores the acceptable string identifiers for learning rate schedule.
    """
    # will be support item for future.
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class OptimizerType(BaseEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """
    # supported item for test, will be delete in the future.
    ADAMWEIGHTDECAY = 'AdamWeightDecay'

    # will be support item for future.
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAFACTOR = "adafactor"


class WrapperType(BaseEnum):
    """
    Stores the acceptable string identifiers for training wrapper.
    """
    # will be support item for future.
    MFWRAPPER = 'mf_wrapper'
    TRAINONESTEP = 'wrapper'
    TRAINONESTEPWITHLOSSSCALE = 'loss_scale_wrapper'


def set_seed(seed: int = 0):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `MindSpore`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    ms_set_seed(seed)


def check_keywords_in_name(name, keywords=()):
    """ Check keywords in name. """
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def check_runner_config(config, dataset):
    """ Check runner config. """
    data_size = dataset.get_dataset_size()
    new_epochs = config.runner_config.epochs
    config.runner_config.origin_epochs = new_epochs
    if config.runner_config.initial_epoch is None:
        config.runner_config.initial_epoch = 0
    if config.runner_config.sink_mode:
        if config.runner_config.sink_size != -1:
            if config.runner_config.sink_size <= 0:
                raise ValueError("per epoch size must be more than 0 or equal to -1, "
                                 f"but get {config.runner_config.sink_size}")
            if data_size < config.runner_config.sink_size:
                logger.warning("The data size %s (get from dataset.get_dataset_size()) is smaller "
                               "than the sink_size %s (get from config.runner_config.sink_size), "
                               "you should set the config.runner_config.sink_size to %s",
                               data_size, config.runner_config.sink_size, data_size)
            config.runner_config.epochs = int((data_size / config.runner_config.sink_size) * new_epochs)
        else:
            config.runner_config.sink_size = data_size
    else:
        logger.warning("Sink mode is False, per epoch size is invalid, it will reset -1.")
        config.runner_config.sink_size = -1

    config.data_size = data_size
    logger.info("Will be Training epochs:%d, sink_size:%d",
                config.runner_config.origin_epochs, config.runner_config.sink_size)
    logger.info("Create training dataset finish, dataset size:%d", data_size)


def check_train_data_loader_type(new_config, old_config):
    """Check train data loader config type."""
    if new_config.train_dataset is None:
        return None
    if new_config.train_dataset.get('data_loader') is None:
        return None
    train_data_loader_type = new_config.train_dataset.get('data_loader').get('type')
    if old_config.train_dataset is not None and train_data_loader_type is not None:
        default_train_data_loader_type = old_config.train_dataset.data_loader.type
        if train_data_loader_type != default_train_data_loader_type:
            logger.warning("train dataset's data_loader type is changed to %s."
                           "The default parameters will be cleared."
                           "Please make sure to input the corresponding parameter values manually.",
                           train_data_loader_type)
            old_config.train_dataset.data_loader = {}
    return None


def check_eval_data_loader_type(new_config, old_config):
    """Check eval data loader config type."""
    if new_config.eval_dataset is None:
        return None
    if new_config.eval_dataset.get('data_loader') is None:
        return None
    eval_data_loader_type = new_config.eval_dataset.get('data_loader').get('type')
    if old_config.eval_dataset is not None and eval_data_loader_type is not None:
        default_eval_data_loader_type = old_config.eval_dataset.data_loader.type
        if eval_data_loader_type != default_eval_data_loader_type:
            logger.warning("eval dataset's data_loader type is changed to %s."
                           "The default parameters will be cleared."
                           "Please make sure to input the corresponding parameter values manually.",
                           eval_data_loader_type)
            old_config.eval_dataset.data_loader = {}
    return None


def check_optimizer_and_lr_type(new_config, old_config):
    """Check optimizer and lr schedule config type."""
    optimizer_type = new_config.optimizer.get('type')
    if old_config.optimizer is not None and optimizer_type is not None:
        default_optimizer_type = old_config.optimizer.type
        if optimizer_type != default_optimizer_type:
            logger.warning(
                "optimizer type is changed to %s."
                "The default parameters will be cleared."
                "Please make sure to input the corresponding parameter values manually except (params).",
                optimizer_type)
            old_config.optimizer = {}

    if hasattr(new_config.optimizer, 'learning_rate'):
        lr_type = new_config.optimizer.learning_rate.get('type')
        if old_config.lr_schedule is not None and lr_type is not None:
            default_lr_type = old_config.lr_schedule.type
            if lr_type != default_lr_type:
                logger.warning(
                    "lr schedule type is changed to %s."
                    "The default parameters will be cleared."
                    "Please make sure to input the corresponding parameter values manually.",
                    lr_type)
                old_config.lr_schedule = None


def check_wrapper_config(new_config, old_config):
    """Check wrapper config."""
    wrapper_type = new_config.runner_wrapper.get('type')
    if old_config.runner_wrapper is not None and wrapper_type is not None:
        default_wrapper_type = old_config.runner_wrapper.type
        if wrapper_type != default_wrapper_type:
            logger.warning(
                "wrapper type is changed to %s."
                "The default parameters will be cleared."
                "Please make sure to input the corresponding parameter values manually.",
                wrapper_type)
            old_config.runner_wrapper = {}


def config2dict(config):
    """MindFormerConfig Type Convert to Dict."""
    if not isinstance(config, (dict, MindFormerConfig)):
        return config
    new_dict = {}
    for key, value in config.items():
        if isinstance(value, MindFormerConfig):
            value = config2dict(value)
        new_dict.setdefault(key, value)
    return new_dict


def load_distributed_checkpoint(config, specify_prefix=None):
    """Load Checkpoint in Parallel Mode."""
    checkpoint_dir = config.load_checkpoint
    if os.path.isdir(checkpoint_dir):
        logger.info(
            "When distributed loads are sliced weights,"
            "load_checkpoint should be a checkpoint directory containing the directory of rank_{0-*},"
            "The directory structure is as follows: **checkpoint_root_dir/checkpoint/rank_{0-*}/**.ckpt")
        distribute_checkpoint_dir = os.path.join(
            checkpoint_dir, "rank_{}".format(int(os.getenv("RANK_ID", "0"))))
        distribute_checkpoint_path = get_last_checkpoint(distribute_checkpoint_dir)
    elif os.path.isfile(checkpoint_dir):
        logger.info("Your load_checkpoint is file, it will be load in network.")
        distribute_checkpoint_path = checkpoint_dir
    else:
        raise FileNotFoundError(f"{checkpoint_dir} is not found.")
    checkpoint_dict = load_checkpoint(distribute_checkpoint_path, specify_prefix=specify_prefix)
    logger.info("Distribute load is success.")
    return checkpoint_dict


def load_resume_context_from_checkpoint(config):
    """resume training, load training info from checkpoint to config"""
    if not os.path.realpath(config.load_checkpoint) or \
            not os.path.exists(config.load_checkpoint):
        raise FileNotFoundError(f"The load_checkpoint must be correct, "
                                f"but get {config.load_checkpoint}")

    if os.path.isdir(config.load_checkpoint):
        resume_dict = load_distributed_checkpoint(config, ["loss_scale", "epoch_num"])
        if not config.runner_config.sink_mode:
            raise ValueError("When distributed loads are sliced weights, sink_mode must be set True.")
    else:
        resume_dict = load_checkpoint(config.load_checkpoint, specify_prefix=["loss_scale", "epoch_num"])

    if "epoch_num" in resume_dict:
        config.runner_config.initial_epoch = int(resume_dict["epoch_num"])
    else:
        config.runner_config.initial_epoch = 0

    for callback in config.callbacks:
        if "type" in callback and callback["type"] == "CheckpointMointor":
            if config.runner_wrapper.scale_sense is not None and "loss_scale" in resume_dict:
                config.runner_wrapper.scale_sense.loss_scale_value = resume_dict["loss_scale"]
            break


def transform_and_load_checkpoint(config, model, network, dataset, optimizer=None, do_eval=False, do_predict=False):
    """
    load checkpoint into net, transform checkpoint if transform is True
    1. build net if parallel mode is auto_parallel
    2. get strategy
    3. transform checkpoint if need
    4. load ckpt
    """
    if not config.only_save_strategy and (not os.path.realpath(config.load_checkpoint) or
                                          not os.path.exists(config.load_checkpoint)):
        raise FileNotFoundError(f"The load_checkpoint must be correct, "
                                f"but get {config.load_checkpoint}")

    if context.get_auto_parallel_context('parallel_mode') in ['semi_auto_parallel', 'auto_parallel',
                                                              'hybrid_parallel']:
        # 1. build net if parallel mode is auto_parallel
        build_model(config, model, dataset, do_eval=do_eval, do_predict=do_predict)
        if config.only_save_strategy:
            logger.info("Only_save_strategy is True, model.compile() finished, system exit! ")
            sys.exit(0)
    elif config.only_save_strategy:
        logger.info("only_save_strategy is True, "
                    "but stand_alone and data_parallel mode do not have strategy file, system exit! ")
        sys.exit(0)

    if config.auto_trans_ckpt:
        # 2. get strategy
        src_ckpt_strategy = get_strategy(config.src_strategy_path_or_dir)
        dst_ckpt_strategy = get_dst_strategy(config)
        # 3. transform checkpoint if needed
        transform_ckpt(config, src_ckpt_strategy=src_ckpt_strategy, dst_ckpt_strategy=dst_ckpt_strategy)

    # 4. load ckpt
    load_ckpt(config, network, optimizer=optimizer)


def build_model(config, model, dataset, do_eval=False, do_predict=False):
    """build model, generate strategy file"""
    if context.get_auto_parallel_context('parallel_mode') in ['semi_auto_parallel', 'auto_parallel',
                                                              'hybrid_parallel']:
        if not config.runner_config.sink_mode:
            raise ValueError("When distributed loads are sliced weights, sink_mode must be set True.")
        if do_eval:
            model.infer_predict_layout(*next(dataset.create_tuple_iterator()))
        elif do_predict:
            model.infer_predict_layout(dataset)
        else:
            if config.runner_config.epochs > 1 and config.runner_config.sink_size == 1:
                raise ValueError(f"When distributed loads are sliced weights, it does not support"
                                 f"epochs = {config.runner_config.epochs} > 1 and "
                                 f"sink_size = {config.runner_config.sink_size} = 1,"
                                 f"sink_size must be more than 1")
            logger.info(".........Building model.........")
            model.build(train_dataset=dataset, epoch=config.runner_config.epochs,
                        sink_size=config.runner_config.sink_size)


def get_dst_strategy(config):
    """get strategy"""
    rank_id = int(os.getenv('RANK_ID', '0'))
    world_size = int(os.getenv('RANK_SIZE', '1'))
    dst_strategy_path = None
    if world_size == 1:
        return dst_strategy_path

    if check_in_modelarts():
        # local send all strategy file to obs
        obs_save_dir = os.path.join(config.remote_save_url, "strategy")
        mox.file.make_dirs(obs_save_dir)
        local_strategy_path = config.parallel.strategy_ckpt_save_file
        obs_strategy_path = os.path.join(obs_save_dir, os.path.basename(local_strategy_path))
        mox.file.copy(local_strategy_path, obs_strategy_path)

        # obs send all strategy to each node
        logger.info(".........Collecting strategy.........")
        local_strategy_dir = os.path.join(get_output_root_path(), "strategy")
        if rank_id % 8 == 0:
            wait_collect_all_strategy(local_strategy_dir, world_size, obs_save_dir)
        else:
            wait_collect_all_strategy(local_strategy_dir, world_size)

        logger.info(".........All strategy as follow.........")
        local_strategy_paths = glob(os.path.join(local_strategy_dir, "*_rank_*.ckpt"))
        local_strategy_paths.sort()
        for local_strategy_path in local_strategy_paths:
            logger.info("strategy: %s", local_strategy_path)
        logger.info(".........Collecting %d strategy.........", len(local_strategy_paths))

        # merge strategy if pipeline_stage > 1
        if config.parallel_config.pipeline_stage > 1:
            if rank_id % 8 == 0:
                logger.info(".........Merging strategy.........")
                merged_strategy_path = get_strategy(local_strategy_dir)
                merged_strategy_name = os.path.basename(merged_strategy_path)
                obs_merged_strategy_path = os.path.join(obs_save_dir, merged_strategy_name)
                mox.file.copy(merged_strategy_path, obs_merged_strategy_path)
                logger.info("Save %s to %s", merged_strategy_path, obs_merged_strategy_path)
                logger.info(".........Merging succeed.........")
                dst_strategy_path = merged_strategy_path
            else:
                dst_strategy_path = None
        else:
            dst_strategy_path = local_strategy_path
    else:
        logger.info(".........Collecting strategy.........")
        local_strategy_dir = os.path.join(get_output_root_path(), "strategy")
        if world_size <= 8:
            wait_collect_all_strategy(local_strategy_dir, world_size)

            logger.info(".........All strategy as follow.........")
            local_strategy_paths = glob(os.path.join(local_strategy_dir, "*_rank_*.ckpt"))
            local_strategy_paths.sort()
            for local_strategy_path in local_strategy_paths:
                logger.info("strategy: %s", local_strategy_path)
            logger.info(".........Collecting %d strategy.........", len(local_strategy_paths))

            # merge strategy if pipeline_stage > 1
            if config.parallel_config.pipeline_stage > 1:
                if rank_id % 8 == 0:
                    logger.info(".........Merging strategy.........")
                    merged_strategy_path = get_strategy(local_strategy_dir)
                    logger.info(".........Merging succeed.........")
                    dst_strategy_path = merged_strategy_path
                else:
                    dst_strategy_path = None
            else:
                dst_strategy_path = local_strategy_paths[0]
        else:
            logger.warning("Can't collecting all strategy, device num > 8!")
            config.auto_trans_ckpt = False
            dst_strategy_path = None

    return dst_strategy_path


def transform_ckpt(config, src_ckpt_strategy=None, dst_ckpt_strategy=None):
    """auto transform ckpt"""
    if not config.load_checkpoint or not os.path.exists(config.load_checkpoint):
        raise ValueError("The load_checkpoint should be exist, "
                         f"but get {config.load_checkpoint}.")
    if not os.path.isdir(config.load_checkpoint) or \
        not glob(os.path.join(config.load_checkpoint, "rank*")):
        raise ValueError("The load_checkpoint must be a dir and "
                         "ckpt should be stored in the format of load_checkpoint/rank_x/xxx.ckpt,"
                         f"but get {config.load_checkpoint}.")

    rank_id = get_rank() if config.use_parallel else 0
    world_size = get_group_size() if config.use_parallel else 1
    transformed_ckpt_dir = os.path.join(get_output_root_path(), "transformed_checkpoint")
    os.makedirs(transformed_ckpt_dir, exist_ok=True)
    if rank_id % 8 == 0:
        logger.info(".........Transforming ckpt.........")
        logger.info("Src ckpt strategy: %s", src_ckpt_strategy)
        logger.info("Src ckpt: %s", config.load_checkpoint)
        logger.info("Dst ckpt strategy: %s", dst_ckpt_strategy)
        logger.info("Dst ckpt: %s", transformed_ckpt_dir)
        try:
            ms.transform_checkpoints(config.load_checkpoint,
                                     transformed_ckpt_dir,
                                     'checkpoint_',
                                     src_ckpt_strategy,
                                     dst_ckpt_strategy)
            logger.info(".........Transform succeed!.........")
            transform_succeed_txt = os.path.join(transformed_ckpt_dir,
                                                 f'transform_succeed_rank_{rank_id}.txt')
            f = open(transform_succeed_txt, 'w')
            f.close()
        except RuntimeError:
            logger.error(".........Transform failed!.........")
            transform_failed_txt = os.path.join(transformed_ckpt_dir,
                                                f'transform_failed_rank_{rank_id}.txt')
            f = open(transform_failed_txt, 'w')
            f.close()

        if check_in_modelarts():
            transformed_ckpt_dir_obs = os.path.join(config.remote_save_url, "transformed_checkpoint")
            mox.file.make_dirs(transformed_ckpt_dir_obs)

            transform_succeed_txt = os.path.join(transformed_ckpt_dir,
                                                 f'transform_succeed_rank_{rank_id}.txt')
            if os.path.exists(transform_succeed_txt):
                transform_succeed_txt_obs = os.path.join(transformed_ckpt_dir_obs,
                                                         f'transform_succeed_rank_{rank_id}.txt')
                mox.file.copy(transform_succeed_txt, transform_succeed_txt_obs)

            transform_failed_txt = os.path.join(transformed_ckpt_dir,
                                                f'transform_failed_rank_{rank_id}.txt')
            if os.path.exists(transform_failed_txt):
                transform_failed_txt_obs = os.path.join(transformed_ckpt_dir_obs,
                                                        f'transform_failed_rank_{rank_id}.txt')
                mox.file.copy(transform_failed_txt, transform_failed_txt_obs)

    wait_transform(config, world_size)

    if check_in_modelarts():
        transformed_ckpt_dir_obs = os.path.join(config.remote_save_url, "transformed_checkpoint")
        rank_transformed_ckpt_dir_obs = os.path.join(transformed_ckpt_dir_obs, f'rank_{rank_id}')
        rank_transformed_ckpt_dir = os.path.join(get_output_root_path(), "transformed_checkpoint", f'rank_{rank_id}')
        mox_adapter(rank_transformed_ckpt_dir, rank_transformed_ckpt_dir_obs)
        print(f"Rank {rank_id}: Save {rank_transformed_ckpt_dir} to {rank_transformed_ckpt_dir_obs}")

    config.load_checkpoint = transformed_ckpt_dir


def wait_transform(config, world_size):
    """wait all node transform over"""
    last_count = -1
    if check_in_modelarts():
        transformed_ckpt_dir_obs = os.path.join(config.remote_save_url, "transformed_checkpoint")
        while True:
            transform_failed_txts_obs = mox.file.glob(os.path.join(transformed_ckpt_dir_obs,
                                                                   f'transform_failed_rank_*.txt'))
            if transform_failed_txts_obs:
                raise ValueError(f"Transform failed, find {transform_failed_txts_obs}.")

            transform_succeed_txts_obs = mox.file.glob(os.path.join(transformed_ckpt_dir_obs,
                                                                    f'transform_succeed_rank_*.txt'))
            current_count = len(transform_succeed_txts_obs)
            total_num = len(list(range(0, world_size, 8)))
            progress = (current_count / total_num) * 100
            if current_count != last_count:
                show_progress(progress, prefix="Transforming checkpoint")
                last_count = current_count
            if current_count < total_num:
                time.sleep(5)
            else:
                break
    else:
        transformed_ckpt_dir = os.path.join(get_output_root_path(), "transformed_checkpoint")
        while True:
            transform_failed_txts = glob(os.path.join(transformed_ckpt_dir,
                                                      f'transform_failed_rank_*.txt'))
            if transform_failed_txts:
                raise ValueError(f"Transform failed, find {transform_failed_txts}.")

            transform_succeed_txts = glob(os.path.join(transformed_ckpt_dir,
                                                       f'transform_succeed_rank_*.txt'))
            current_count = len(transform_succeed_txts)
            total_num = len(list(range(0, world_size, 8)))
            progress = (current_count / total_num) * 100
            if current_count != last_count:
                show_progress(progress, prefix="Transforming checkpoint")
                last_count = current_count
            if current_count < total_num:
                time.sleep(5)
            else:
                break


def wait_collect_all_strategy(strategy_dir, total_num, obs_strategy_dir=None):
    """wait all strategy collect over"""
    last_count = -1
    last_count_obs = -1
    while True:
        if obs_strategy_dir:
            obs_strategy_paths = mox.file.glob(os.path.join(obs_strategy_dir, "*.ckpt"))
            obs_current_count = len(obs_strategy_paths)
            progress = (obs_current_count / total_num) * 100
            if obs_current_count != last_count_obs:
                show_progress(progress, prefix="Collecting strategy")
                last_count_obs = obs_current_count
            if obs_current_count < total_num:
                time.sleep(5)
                continue
            for obs_strategy_path in obs_strategy_paths:
                local_strategy_path = os.path.join(strategy_dir, os.path.basename(obs_strategy_path))
                mox.file.copy(obs_strategy_path, local_strategy_path)

        local_strategy_paths = glob(os.path.join(strategy_dir, "*_rank_*.ckpt"))
        local_current_count = len(local_strategy_paths)
        progress = (local_current_count / total_num) * 100
        if local_current_count != last_count:
            show_progress(progress, prefix="Collecting strategy")
            last_count = local_current_count
        if local_current_count < total_num:
            time.sleep(5)
        else:
            break


def show_progress(progress, prefix=''):
    """show progress"""
    show_str = ('|%%-%ds|' % 50) % (int(50 * progress / 100) * "▮")
    logger.info("%s: %s%d%%", prefix, show_str, progress)


def load_ckpt(config, network, optimizer=None):
    """load checkpoint"""
    # load checkpoint params into dict
    logger.info(".............Start load checkpoint from checkpoint..................")
    if os.path.isdir(config.load_checkpoint):
        checkpoint_dict = load_distributed_checkpoint(config)
    else:
        checkpoint_dict = load_checkpoint(config.load_checkpoint)

    # replace tk in checkpoint_dict.keys()
    checkpoint_dict = replace_tk_to_mindpet(checkpoint_dict)

    # load params into net
    not_load_network_params = load_param_into_net(network, checkpoint_dict)
    logger.info("Network parameters are not loaded: %s", str(not_load_network_params))
    if optimizer:
        not_load_optim_params = load_param_into_net(optimizer, checkpoint_dict)
        logger.info("Optimizer parameters are not loaded: %s", str(not_load_optim_params))


def get_last_checkpoint(checkpoint_dir):
    """get last checkpoint for resuming or finetune."""
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(
            f"{checkpoint_dir} is not a real directory,"
            "When distributed loads are sliced weights,"
            "load_checkpoint should be a checkpoint directory containing the directory of rank_{0-*},"
            "The directory structure is as follows: **checkpoint_root_dir/rank_{0-*}/checkpoint/**.ckpt")
    output_checkpoint_path = [
        checkpoint for checkpoint in os.listdir(checkpoint_dir)
        if checkpoint.endswith('.ckpt')
    ]
    if not output_checkpoint_path:
        return None
    output_checkpoint_path = sorted(output_checkpoint_path,
                                    key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, output_checkpoint_path[-1])
