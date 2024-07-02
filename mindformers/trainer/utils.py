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
from enum import Enum

import numpy as np

from mindspore import context, load_checkpoint, load_param_into_net
from mindspore import set_seed as ms_set_seed

from mindformers.tools.logger import logger
from mindformers.tools.utils import get_real_rank
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.utils import (
    check_in_modelarts,
    get_output_root_path,
    replace_tk_to_mindpet,
    check_shared_disk,
    get_remote_save_url,
    remove_folder,
    get_device_num_per_node,
    format_path
)
from mindformers.tools.ckpt_transform import TransformCkpt


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

    if os.path.isdir(config.load_checkpoint) and not isinstance(config.resume_training, str):
        resume_dict = load_distributed_checkpoint(config.load_checkpoint,
                                                  choice_func=lambda x: x in ["loss_scale", "epoch_num", "step_num"])
    else:
        if isinstance(config.resume_training, str):
            checkpoint_tmp = os.path.join(config.load_checkpoint, f"rank_{config.rank_id}", config.resume_training)
        else:
            checkpoint_tmp = config.load_checkpoint
        resume_dict = load_checkpoint(checkpoint_tmp,
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
        if "type" in callback and callback["type"] == "CheckpointMonitor":
            if config.runner_wrapper.scale_sense is not None and "loss_scale" in resume_dict:
                if hasattr(config.runner_wrapper.scale_sense, "loss_scale_value"):
                    config.runner_wrapper.scale_sense.loss_scale_value = resume_dict["loss_scale"]
                else:
                    config.runner_wrapper.scale_sense = resume_dict["loss_scale"]
            break


def delete_resume_record_dir(wait_time=5):
    """delete resume record dir"""
    if check_in_modelarts():
        resume_record_dir = os.path.join(get_remote_save_url(), "resume_record")
    else:
        resume_record_dir = os.path.join(get_output_root_path(), "resume_record")
    time.sleep(wait_time)
    remove_folder(resume_record_dir)


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

    if not config.auto_trans_ckpt and not config.only_save_strategy and do_predict:
        network.set_train(False)
        load_ckpt(config, network)
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
        logger.info("%s is_share_disk: %r", os.path.abspath(config.output_dir), is_share_disk)
        transform_process_num = config.transform_process_num if config.transform_process_num else 1
        transform_by_rank = config.transform_by_rank if config.transform_by_rank else False
        npu_num_per_node = config.npu_num_per_node \
            if config.npu_num_per_node else get_device_num_per_node()
        transform_ckpt = TransformCkpt(auto_trans_ckpt=True,
                                       transform_process_num=transform_process_num,
                                       transform_by_rank=transform_by_rank,
                                       npu_num_per_node=npu_num_per_node)
        config.load_checkpoint = transform_ckpt(
            src_checkpoint=config.load_checkpoint,
            src_strategy=config.src_strategy_path_or_dir
        )

    # 5. load ckpt
    load_ckpt(config, network, optimizer=optimizer)


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


def build_model(config, model, dataset, do_eval=False, do_predict=False):
    """build model and generate strategy file"""
    if context.get_auto_parallel_context('parallel_mode') in ['semi_auto_parallel', 'auto_parallel',
                                                              'hybrid_parallel']:
        if not config.runner_config.sink_mode:
            raise ValueError("When distributed loads are sliced weights, sink_mode must be set True.")
        if do_eval:
            model.infer_predict_layout(*dataset)
        elif do_predict:
            model.infer_predict_layout(*dataset)
        else:
            model.build(train_dataset=dataset, epoch=config.runner_config.epochs,
                        sink_size=config.runner_config.sink_size)


def load_ckpt(config, network, optimizer=None):
    """load checkpoint"""
    # load checkpoint params into dict
    logger.info("............Start load checkpoint from checkpoint............")
    checkpoint_dict = {}
    rank_id = get_real_rank() if get_real_rank() else 0
    config.load_checkpoint = format_path(config.load_checkpoint)
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
            if isinstance(config.resume_training, str):
                checkpoint_tmp = os.path.join(config.load_checkpoint, f"rank_{config.rank_id}", config.resume_training)
                checkpoint_dict = load_checkpoint(checkpoint_tmp)
            elif config.use_parallel:
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
