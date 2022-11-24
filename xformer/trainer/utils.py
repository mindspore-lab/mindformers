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
        config.runner_config.epochs = int((data_size / config.runner_config.per_epoch_size) * new_epochs)
    else:
        config.runner_config.per_epoch_size = data_size

    config.data_size = data_size
    config.logger.info("Will be Training epochs:{}, sink_size:{}".format(
        config.runner_config.epochs, config.runner_config.per_epoch_size))
    config.logger.info("Create training dataset finish, dataset size:{}".format(data_size))
