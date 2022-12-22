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

from mindformers.tools.logger import logger


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
    if config.runner_config.per_epoch_size and config.runner_config.sink_mode:
        if data_size < config.runner_config.per_epoch_size:
            logger.warning("The data size %s (get from dataset.get_dataset_size()) is smaller "
                           "than the per_epoch_size %s (get from config.runner_config.per_epoch_size), "
                           "you should set the config.runner_config.per_epoch_size to %s",
                           data_size, config.runner_config.per_epoch_size, data_size)
        config.runner_config.epochs = int((data_size / config.runner_config.per_epoch_size) * new_epochs)
    else:
        config.runner_config.per_epoch_size = data_size

    config.data_size = data_size
    logger.info("Will be Training epochs:%d, sink_size:%d",
                config.runner_config.epochs, config.runner_config.per_epoch_size)
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


def check_lr_config(new_config, old_config):
    """Check lr schedule config."""
    lr_type = new_config.lr_schedule.type
    if old_config.lr_schedule is not None and lr_type is not None:
        default_lr_type = old_config.lr_schedule.type
        if lr_type != default_lr_type:
            logger.warning(
                "lr schedule type is changed to %s."
                "The default parameters will be cleared."
                "Please make sure to input the corresponding parameter values manually.",
                lr_type)
            old_config.lr_schedule = None


def _check_lr_config(config, device_num=1, batch_size=128, arch="simmim"):
    if arch in ('simmim', "ringmo", "ringmo_mm"):
        config.base_lr = (config.base_lr * device_num * batch_size) / 512
        config.min_lr = (config.min_lr * device_num * batch_size) / 512
        config.warmup_lr = (config.warmup_lr * device_num * batch_size) / 512
    if arch in ('MaeModel', 'VitModel'):
        config.base_lr = (config.base_lr * device_num * batch_size) / 256


def check_image_lr_config(config):
    """config lr"""
    lr_config = config.lr_schedule
    device_num = config.device_num
    batch_size = config.runner_config.batch_size
    _check_lr_config(lr_config, device_num=device_num, batch_size=batch_size, arch=config.model.arch.type)
    total_epochs = config.runner_config.epochs
    steps_per_epoch = config.data_size
    if config.runner_config.per_epoch_size and config.runner_config.sink_mode:
        total_steps = total_epochs
    else:
        total_steps = total_epochs * steps_per_epoch
    lr_config.warmup_steps = int(lr_config.warmup_epochs * steps_per_epoch)
    lr_config.decay_steps = total_steps - lr_config.warmup_steps
    del lr_config.warmup_epochs


def check_model_config(config):
    config.model.parallel_config = config.parallel_config
    config.model.moe_config = config.moe_config
    config.model.batch_size = config.runner_config.batch_size * config.device_num \
        if config.parallel.parallel_mode == "semi_auto_parallel" else config.runner_config.batch_size
    config.model.image_size = config.runner_config.image_size
    config.model.num_classes = config.runner_config.num_classes


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
