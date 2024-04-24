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
import shutil
from glob import glob
from enum import Enum

import numpy as np

import mindspore as ms
from mindspore.communication.management import get_rank, get_group_size
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore import set_seed as ms_set_seed

from mindformers.tools.logger import logger
from mindformers.tools.utils import get_real_rank, get_real_group_size
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.utils import check_in_modelarts, get_output_root_path, \
                                    replace_tk_to_mindpet, check_shared_disk
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


class IntervalStrategy(BaseEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class HubStrategy(BaseEnum):
    END = "end"
    EVERY_SAVE = "every_save"
    CHECKPOINT = "checkpoint"
    ALL_CHECKPOINTS = "all_checkpoints"


class LoggingIntervalStrategy(BaseEnum):
    STEPS = "steps"
    EPOCH = "epoch"


class SaveIntervalStrategy(BaseEnum):
    """
    Stores the acceptable string identifiers for save checkpoint monitor.
    """
    NO = "no"
    STEPS = "steps"
    SECONDS = "seconds"


class LrSchedulerType(BaseEnum):
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
    FUSED_ADAMW = "fused_adamw"
    FP32_ADAMW = "fp32_adamw"


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
    if config.runner_config.gradient_accumulation_steps is None:
        config.runner_config.gradient_accumulation_steps = 1
    if config.runner_config.initial_epoch is None:
        config.runner_config.initial_epoch = 0
    if config.runner_config.initial_step is None:
        config.runner_config.initial_step = 0
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


def load_distributed_checkpoint(checkpoint_dir, choice_func=None):
    """Load Checkpoint in Parallel Mode."""
    if os.path.isdir(checkpoint_dir):
        logger.info(
            "When distributed loads are sliced weights,"
            "load_checkpoint should be a checkpoint directory containing the directory of rank_{0-*},"
            "The directory structure is as follows: **checkpoint_root_dir/rank_{0-*}/**.ckpt")
        distribute_checkpoint_dir = os.path.join(
            checkpoint_dir, "rank_{}".format(get_real_rank()))
        distribute_checkpoint_path = get_last_checkpoint(distribute_checkpoint_dir)
    elif os.path.isfile(checkpoint_dir):
        logger.info("Your load_checkpoint is file, it will be load in network.")
        distribute_checkpoint_path = checkpoint_dir
    else:
        raise FileNotFoundError(f"{checkpoint_dir} is not found.")
    checkpoint_dict = load_checkpoint(distribute_checkpoint_path, choice_func=choice_func)
    logger.info("Distribute load is success.")
    return checkpoint_dict


def load_resume_context_from_checkpoint(config, dataset):
    """resume training, load training info from checkpoint to config"""
    if not os.path.realpath(config.load_checkpoint) or \
            not os.path.exists(config.load_checkpoint):
        raise FileNotFoundError(f"The load_checkpoint must be correct, "
                                f"but get {config.load_checkpoint}")

    if os.path.isdir(config.load_checkpoint):
        resume_dict = load_distributed_checkpoint(config.load_checkpoint,
                                                  choice_func=lambda x: x in ["loss_scale", "epoch_num", "step_num"])
    else:
        resume_dict = load_checkpoint(config.load_checkpoint,
                                      choice_func=lambda x: x in ["loss_scale", "epoch_num", "step_num"])

    if "step_num" in resume_dict:
        config.runner_config.initial_step = int(resume_dict["step_num"])
    else:
        config.runner_config.initial_step = 0

    if "epoch_num" in resume_dict:
        if config.runner_config.sink_mode:
            config.runner_config.initial_epoch = int(resume_dict["epoch_num"])
        else:
            data_size = dataset.get_dataset_size()
            not_last_step_in_epoch = int(config.runner_config.initial_step % data_size != 0)
            config.runner_config.initial_epoch = int(resume_dict["epoch_num"]) - not_last_step_in_epoch
    else:
        config.runner_config.initial_epoch = 0

    for callback in config.callbacks:
        if "type" in callback and callback["type"] == "CheckpointMointor":
            if config.runner_wrapper.scale_sense is not None and "loss_scale" in resume_dict:
                if hasattr(config.runner_wrapper.scale_sense, "loss_scale_value"):
                    config.runner_wrapper.scale_sense.loss_scale_value = resume_dict["loss_scale"]
                else:
                    config.runner_wrapper.scale_sense = resume_dict["loss_scale"]
            break


def transform_and_load_checkpoint(config, model, network, dataset, optimizer=None, do_eval=False, do_predict=False):
    """
    load checkpoint into net, transform checkpoint if transform is True
    1. build net if parallel mode is auto_parallel
    2. get strategy
    3. make softlink of input path
    4. transform checkpoint if need
    5. load ckpt
    """
    if not config.only_save_strategy and (not os.path.realpath(config.load_checkpoint) or
                                          not os.path.exists(config.load_checkpoint)):
        raise FileNotFoundError(f"The load_checkpoint must be correct, "
                                f"but get {config.load_checkpoint}")

    if not config.auto_trans_ckpt and not config.only_save_strategy and \
        check_path_include_total_ckpt(config.load_checkpoint):
        load_ckpt(config, network, optimizer=optimizer)
        return

    if context.get_auto_parallel_context('parallel_mode') in ['semi_auto_parallel', 'auto_parallel',
                                                              'hybrid_parallel']:
        # 1. build net if parallel mode is auto_parallel
        logger.info(".........Building model.........")
        build_model(config, model, dataset, do_eval=do_eval, do_predict=do_predict)
        if config.only_save_strategy:
            logger.info("Only_save_strategy is True, model.compile() finished, system exit! ")
            sys.exit(0)
    elif config.only_save_strategy:
        logger.info("only_save_strategy is True, "
                    "but stand_alone and data_parallel mode do not have strategy file, system exit! ")
        sys.exit(0)

    if config.auto_trans_ckpt:
        is_share_disk = check_shared_disk(config.output_dir)
        world_size = int(os.getenv('RANK_SIZE', '1'))
        logger.info("%s is_share_disk: %r", os.path.abspath(config.output_dir), is_share_disk)
        logger.info("world_size: %d", world_size)

        # 2. get strategy
        src_ckpt_strategy, dst_ckpt_strategy = get_src_and_dst_strategy(config)

        # 3. check format of input path and make softlink
        softlink_dir = check_ckpt_for_transform(config.load_checkpoint)

        # 4. transform checkpoint if needed
        for ckpt_dir in os.listdir(softlink_dir):
            config.load_checkpoint = os.path.join(softlink_dir, ckpt_dir)
            transform_ckpt(config, ckpt_dir,
                           src_ckpt_strategy=src_ckpt_strategy,
                           dst_ckpt_strategy=dst_ckpt_strategy)

    # 5. load ckpt
    load_ckpt(config, network, optimizer=optimizer)


def check_ckpt_for_transform(ckpt_dir):
    """check input ckpt_dir and transform it by using softlink"""
    soft_link_dir = os.path.join(get_output_root_path(), "softlink_ckpt")
    rank_id = get_real_rank()

    if os.path.isdir(ckpt_dir) and not check_rank_folders(ckpt_dir, 0) and \
        not check_ckpt_file_exist(ckpt_dir):
        raise ValueError(f"No rank_0 folder or ckpt files are found under {ckpt_dir}.")
    if os.path.isfile(ckpt_dir) and not ckpt_dir.endswith('.ckpt'):
        raise ValueError(f"The value of load_checkpoint must be a folder or a file with suffix '.ckpt', "
                         f"but got {ckpt_dir}")

    if (not rank_id) or (rank_id % 8 == 0 and check_in_modelarts()):
        if os.path.exists(soft_link_dir):
            shutil.rmtree(soft_link_dir)
            logger.info("Find exist softlink dir %s and delete it.", os.path.join(os.getcwd(), soft_link_dir))
        if os.path.isdir(ckpt_dir):
            if check_rank_folders(ckpt_dir, 0):
                if check_ckpt_file_exist(ckpt_dir):
                    logger.warning(f"Find both ckpt files and rank folder under {ckpt_dir}, "
                                   "the rank folder will be used for checkpoint transform.")
                os.makedirs(soft_link_dir, exist_ok=True)
                if ckpt_dir.endswith('/'):
                    ckpt_dir = ckpt_dir[:-1]
                soft_link = os.path.join(soft_link_dir, os.path.basename(ckpt_dir))
                logger.info("Make soft link of checkpoint file from %s to %s", ckpt_dir, soft_link)
                if not os.path.exists(soft_link):
                    os.symlink(ckpt_dir, soft_link)
                else:
                    os.remove(soft_link)
                    os.symlink(ckpt_dir, soft_link)
            else:
                for ckpt_file in os.listdir(ckpt_dir):
                    if ckpt_file.endswith('.ckpt'):
                        soft_link = os.path.join(soft_link_dir, os.path.splitext(ckpt_file)[0])
                        ckpt_file = os.path.join(ckpt_dir, ckpt_file)
                        make_softlink(soft_link, ckpt_file)
        else:
            ckpt_file = ckpt_dir
            soft_link = os.path.join(soft_link_dir, os.path.splitext(os.path.basename(ckpt_file))[0])
            make_softlink(soft_link, ckpt_file)

    wait_create_softlink(soft_link_dir)

    return soft_link_dir


def check_rank_folders(path, rank_id):
    """check if the folders in path are correct"""
    folder_name = "rank_{}".format(rank_id)
    if not os.path.exists(os.path.join(path, folder_name)):
        return False
    return True


def check_ckpt_file_exist(path):
    """check if the files in path endswith .ckpt"""
    for file_name in os.listdir(path):
        if file_name.endswith('.ckpt'):
            return True
    return False


def check_path_include_total_ckpt(path):
    """check if the input path is total, not split."""
    if path is None:
        return False
    if os.path.isdir(path):
        if check_ckpt_file_exist(path):
            return True
    elif path.endswith('.ckpt'):
        return True
    return False


def make_softlink(soft_link_dir, ckpt_file):
    """make softlink to fit format of ms.load_checkpoint"""
    os.makedirs(os.path.join(soft_link_dir, "rank_0"), mode=0o755, exist_ok=True)
    link_name = os.path.join(soft_link_dir, "rank_0", os.path.basename(ckpt_file))
    logger.info("Make soft link of checkpoint file from %s to %s", ckpt_file, link_name)
    if os.path.exists(link_name):
        os.remove(link_name)
    os.symlink(ckpt_file, link_name)


def build_model(config, model, dataset, do_eval=False, do_predict=False):
    """build model, generate strategy file"""
    if context.get_auto_parallel_context('parallel_mode') in ['semi_auto_parallel', 'auto_parallel',
                                                              'hybrid_parallel']:
        if not config.runner_config.sink_mode:
            raise ValueError("When distributed loads are sliced weights, sink_mode must be set True.")
        if do_eval:
            model.infer_predict_layout(*next(dataset.create_tuple_iterator()))
        elif do_predict:
            model.infer_predict_layout(*dataset)
        else:
            model.build(train_dataset=dataset, epoch=config.runner_config.epochs,
                        sink_size=config.runner_config.sink_size)


def get_src_and_dst_strategy(config):
    """get strategy"""
    rank_id = get_real_rank()
    world_size = get_real_group_size()

    if config.src_strategy_path_or_dir:
        assert os.path.exists(config.src_strategy_path_or_dir), \
            f'{config.src_strategy_path_or_dir} not found!'

    dst_strategy_path = None
    if (not rank_id) or (rank_id % 8 == 0 and check_in_modelarts()):
        src_strategy_path = get_strategy(config.src_strategy_path_or_dir)
    else:
        src_strategy_path = None

    if world_size == 1:
        return src_strategy_path, dst_strategy_path

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

        if config.parallel_config.pipeline_stage == 1:
            local_strategy_paths = glob(os.path.join(local_strategy_dir, "*_rank_*.ckpt"))
            local_strategy_paths.sort()
            dst_strategy_path = local_strategy_paths[0]
            logger.info("pipeline_stage = 1, strategy using %s", dst_strategy_path)
            return src_strategy_path, dst_strategy_path

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
        logger.info(".........Collecting strategy.........")
        local_strategy_dir = os.path.join(get_output_root_path(), "strategy")

        if config.parallel_config.pipeline_stage == 1:
            local_strategy_paths = glob(os.path.join(local_strategy_dir, "*_rank_*.ckpt"))
            local_strategy_paths.sort()
            dst_strategy_path = local_strategy_paths[0]
            logger.info("pipeline_stage = 1, strategy using %s", dst_strategy_path)
            return src_strategy_path, dst_strategy_path

        wait_collect_all_strategy(local_strategy_dir, world_size)

        logger.info(".........All strategy as follow.........")
        local_strategy_paths = glob(os.path.join(local_strategy_dir, "*_rank_*.ckpt"))
        local_strategy_paths.sort()
        for local_strategy_path in local_strategy_paths:
            logger.info("strategy: %s", local_strategy_path)
        logger.info(".........Collecting %d strategy.........", len(local_strategy_paths))

        # merge strategy if pipeline_stage > 1
        if not rank_id:
            logger.info(".........Merging strategy.........")
            merged_strategy_path = get_strategy(local_strategy_dir)
            logger.info(".........Merging succeed.........")
            dst_strategy_path = merged_strategy_path
        else:
            dst_strategy_path = None

    return src_strategy_path, dst_strategy_path


def transform_ckpt(config, ckpt_dir, src_ckpt_strategy=None, dst_ckpt_strategy=None):
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
    world_size = get_group_size() if (config.use_parallel and check_in_modelarts()) else 1
    transformed_ckpt_dir = os.path.join(get_output_root_path(), "transformed_checkpoint", ckpt_dir)
    os.makedirs(transformed_ckpt_dir, exist_ok=True)
    if (not rank_id) or (rank_id % 8 == 0 and check_in_modelarts()):
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
        except (NotADirectoryError, TypeError, ValueError, NotImplementedError, RuntimeError) as e:
            logger.error(f".........Transform failed due to: {str(e)}.........")
            transform_failed_txt = os.path.join(transformed_ckpt_dir,
                                                f'transform_failed_rank_{rank_id}.txt')
            f = open(transform_failed_txt, 'w')
            f.close()

        if check_in_modelarts():
            transformed_ckpt_dir_obs = os.path.join(config.remote_save_url, "transformed_checkpoint", ckpt_dir)
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

    wait_transform(config, ckpt_dir, world_size)

    if check_in_modelarts():
        transformed_ckpt_dir_obs = os.path.join(config.remote_save_url, "transformed_checkpoint", ckpt_dir)
        rank_transformed_ckpt_dir_obs = os.path.join(transformed_ckpt_dir_obs, f'rank_{rank_id}')
        rank_transformed_ckpt_dir = os.path.join(get_output_root_path(), "transformed_checkpoint",
                                                 ckpt_dir, f'rank_{rank_id}')
        mox_adapter(rank_transformed_ckpt_dir, rank_transformed_ckpt_dir_obs)
        print(f"Rank {rank_id}: Save {rank_transformed_ckpt_dir} to {rank_transformed_ckpt_dir_obs}")

    config.load_checkpoint = os.path.dirname(transformed_ckpt_dir)


def wait_create_softlink(soft_link_dir):
    """wait softlink create over"""
    while True:
        if os.path.exists(soft_link_dir):
            break
        else:
            time.sleep(5)


def wait_transform(config, ckpt_dir, world_size):
    """wait all node transform over"""
    last_count = -1
    if check_in_modelarts():
        transformed_ckpt_dir_obs = os.path.join(config.remote_save_url, "transformed_checkpoint", ckpt_dir)
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
        transformed_ckpt_dir = os.path.join(get_output_root_path(), "transformed_checkpoint", ckpt_dir)
        while True:
            transform_failed_txts = glob(os.path.join(transformed_ckpt_dir, f'transform_failed_rank_*.txt'))
            if transform_failed_txts:
                raise ValueError(f"Transform failed, find {transform_failed_txts}.")

            transform_succeed_txts = glob(os.path.join(transformed_ckpt_dir, f'transform_succeed_rank_*.txt'))
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
    start_time = time.time()
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
            if time.time() - start_time > 7200:
                raise TimeoutError("Timeout while collecting all strategy!")
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
    logger.info("............Start load checkpoint from checkpoint............")
    checkpoint_dict = {}
    rank_id = get_real_rank() if get_real_rank() else 0
    if config.auto_trans_ckpt:
        for checkpoint_name in os.listdir(config.load_checkpoint):
            checkpoint_path = os.path.join(config.load_checkpoint, checkpoint_name)
            checkpoint_dict.update(load_distributed_checkpoint(checkpoint_path))
            logger.info("loaded checkpoint: %s", str(checkpoint_path))
    else:
        if os.path.isdir(config.load_checkpoint) and check_ckpt_file_exist(config.load_checkpoint):
            for ckpt_file in os.listdir(config.load_checkpoint):
                if ckpt_file.endswith('.ckpt'):
                    checkpoint_path = os.path.join(config.load_checkpoint, ckpt_file)
                    checkpoint_dict.update(load_checkpoint(checkpoint_path))
        elif os.path.isfile(config.load_checkpoint) and config.load_checkpoint.endswith('.ckpt'):
            checkpoint_dict = load_checkpoint(config.load_checkpoint)
        elif os.path.isdir(config.load_checkpoint) and check_rank_folders(config.load_checkpoint, rank_id):
            if config.use_parallel:
                checkpoint_dict = load_distributed_checkpoint(config.load_checkpoint)
            else:
                checkpoint_dict = load_checkpoint(get_last_checkpoint(os.path.join(config.load_checkpoint,
                                                                                   "rank_{}".format(rank_id))))
        else:
            raise ValueError(f"{config.load_checkpoint} is not a valid path to load checkpoint "
                             f"when auto_trans_ckpt is False.")

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
            "The directory structure is as follows: **checkpoint_root_dir/rank_{0-*}/**.ckpt")
    output_checkpoint_path = [
        checkpoint for checkpoint in os.listdir(checkpoint_dir)
        if checkpoint.endswith('.ckpt')
    ]
    if not output_checkpoint_path:
        return None
    output_checkpoint_path = sorted(output_checkpoint_path,
                                    key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, output_checkpoint_path[-1])
