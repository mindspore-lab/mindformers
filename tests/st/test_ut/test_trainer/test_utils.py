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
"""
Test module for testing the trainer interface used for mindformers.
How to run this:
    pytest tests/st/test_trainer/test_utils.py
"""
import tempfile
import os
import pytest

import mindspore as ms

from mindformers.trainer.utils import (check_keywords_in_name, check_train_data_loader_type,
                                       check_eval_data_loader_type, check_optimizer_and_lr_type,
                                       check_wrapper_config, config2dict, MindFormerConfig,
                                       load_distributed_checkpoint, check_rank_folders, check_ckpt_file_exist)

temp_dir = tempfile.TemporaryDirectory()
path = temp_dir.name

class MockConfig:
    """A mock model config class for testing utils."""
    def __init__(self, train_dataset, eval_dataset, optimizer, runner_wrapper):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.runner_wrapper = runner_wrapper

class MockDataLoader:
    """A mock model data loader class for testing utils."""
    def __init__(self):
        self.type = "int"


class MockDataSet:
    """A mock model data set class for testing utils."""
    def __init__(self, data_loader):
        self.data_loader = data_loader

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_keywords_in_name():
    """
    Feature: check_keywords_in_name interface.
    Description: Test basic function of check_keywords_in_name api.
    Expectation: success
    """
    isin = check_keywords_in_name(["test"], ["test"])
    assert isin is True

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_train_data_loader_type():
    """
    Feature: check_train_data_loader_type interface.
    Description: Test basic function of check_train_data_loader_type api.
    Expectation: success
    """
    config = MockConfig(None, None, None, None)
    assert check_train_data_loader_type(config, config) is None

    train_dataset = dict()
    config = MockConfig(train_dataset, None, None, None)
    assert check_train_data_loader_type(config, config) is None

    train_dataset["data_loader"] = dict()
    train_dataset["data_loader"]["type"] = "test"
    config = MockConfig(train_dataset, None, None, None)
    old_train_dataset = MockDataSet(MockDataLoader())

    old_config = MockConfig(old_train_dataset, None, None, None)
    assert check_train_data_loader_type(config, old_config) is None

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_eval_data_loader_type():
    """
    Feature: check_eval_data_loader_type interface.
    Description: Test basic function of check_eval_data_loader_type api.
    Expectation: success
    """
    config = MockConfig(None, None, None, None)
    assert check_eval_data_loader_type(config, config) is None

    eval_dataset = dict()
    config = MockConfig(None, eval_dataset, None, None)
    assert check_eval_data_loader_type(config, config) is None

    eval_dataset["data_loader"] = dict()
    eval_dataset["data_loader"]["type"] = "test"
    config = MockConfig(None, eval_dataset, None, None)
    old_train_dataset = MockDataSet(MockDataLoader())

    old_config = MockConfig(None, old_train_dataset, None, None)
    assert check_eval_data_loader_type(config, old_config) is None

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_optimizer_and_lr_type():
    """
    Feature: check_optimizer_and_lr_type interface.
    Description: Test basic function of check_optimizer_and_lr_type api.
    Expectation: success
    """
    optimizer = dict()
    optimizer["type"] = "test"
    config = MockConfig(None, None, optimizer, None)
    old_config = MockConfig(None, None, MockDataLoader(), None)
    check_optimizer_and_lr_type(config, old_config)
    assert old_config.optimizer == {}

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_wrapper_config():
    """
    Feature: config2dict interface.
    Description: Test basic function of config2dict api.
    Expectation: success
    """
    runner_wrapper = dict()
    runner_wrapper["type"] = "test"
    config = MockConfig(None, None, None, runner_wrapper)
    old_config = MockConfig(None, None, None, MockDataLoader())
    check_wrapper_config(config, old_config)
    assert old_config.runner_wrapper == {}

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_config2dict():
    """
    Feature: config2dict interface.
    Description: Test basic function of config2dict api.
    Expectation: success
    """
    config = MockConfig(None, None, None, MockDataLoader())
    config2dict(config)
    assert isinstance(config, MockConfig)
    value = MindFormerConfig()
    config_ = dict()
    config_["test"] = value
    config2dict(config_)
    assert isinstance(config_, dict)

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_load_distributed_checkpoint():
    """
    Feature: load_distributed_checkpoint interface.
    Description: Test basic function of load_distributed_checkpoint api.
    Expectation: success
    """
    os.mkdir(os.path.join(path, "rank_1"))
    test = dict()
    test["test"] = "test"
    ms.save_checkpoint(test, os.path.join(path, "rank_1/test.ckpt"))
    checkpoint_dict = load_distributed_checkpoint(path, None, 1)
    assert isinstance(checkpoint_dict, dict)
    assert checkpoint_dict["test"] == "test"

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_rank_folders():
    """
    Feature: check_rank_folders interface.
    Description: Test basic function of check_rank_folders api.
    Expectation: success
    """
    judge = check_rank_folders(path, 1)
    assert judge is True

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_ckpt_file_exist():
    """
    Feature: check_ckpt_file_exist interface.
    Description: a testcase for check_ckpt_file_exist
    Expectation: success
    """
    judge = check_ckpt_file_exist(path)
    assert judge is False
