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
"""
Test module for testing the t5 trainer used for mindformers.
How to run this:
pytest tests/st/test_model/test_t5_model/test_t5_trainer.py
"""
import os
import shutil

import numpy as np
import pytest
from mindspore.dataset import MindDataset, GeneratorDataset
from mindspore.mindrecord import FileWriter

from mindformers.trainer import Trainer
from mindformers.trainer.config_args import ConfigArguments, \
    OptimizerConfig, RunnerConfig
from mindformers import T5Config, T5ModelForLoss


def generator(src_length=16, target_length=8):
    """dataset generator"""
    input_ids = np.random.randint(low=0, high=15, size=(src_length,)).astype(np.int32)
    attention_mask = np.ones((src_length,)).astype(np.int32)
    labels = np.random.randint(low=0, high=15, size=(target_length,)).astype(np.int32)

    for _ in range(2):
        yield input_ids, attention_mask, labels


def write_mindrecord(ds_generator, data_record_path):
    """Using the generator to get mindrecords"""
    ds = GeneratorDataset(ds_generator, column_names=["input_ids", "attention_mask", "labels"])

    writer = FileWriter(file_name=data_record_path, shard_num=1, overwrite=True)
    data_schema = {"input_ids": {"type": "int32", "shape": [-1]},
                   "attention_mask": {"type": "int32", "shape": [-1]},
                   "labels": {"type": "int32", "shape": [-1]}}
    writer.add_schema(data_schema, "test_schema")
    for item in ds.create_dict_iterator():
        for k in item.keys():
            item[k] = item[k].asnumpy()
        writer.write_raw_data([item])
    writer.commit()

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_translation_trainer_train_using_common_yaml():
    """
    Feature: Create Trainer From Config
    Description: Test Trainer API to train from config
    Expectation: TypeError
    """
    dir_path = os.path.join(os.path.dirname(__file__), 'fake_dataset')
    os.makedirs(dir_path, exist_ok=True)
    abs_path = os.path.join(dir_path, 't5_dataset')
    write_mindrecord(generator(src_length=16, target_length=8), abs_path)

    batch_size = 1
    runner_config = RunnerConfig(epochs=1, batch_size=batch_size)  # 运行超参
    optim_config = OptimizerConfig(optim_type='AdamWeightDecay', beta1=0.009, learning_rate=0.001)

    dataset_files = []
    for r, _, f in os.walk(dir_path):
        for file in f:
            if not file.endswith("db"):
                dataset_files.append(os.path.join(r, file))
    dataset = MindDataset(dataset_files=dataset_files, columns_list=["input_ids", "attention_mask", "labels"])
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat(1)

    config = ConfigArguments(seed=2022, runner_config=runner_config, optimizer=optim_config)
    model_config = T5Config(batch_size=batch_size, num_heads=8, num_hidden_layers=1, hidden_size=512,
                            seq_length=16, max_decode_length=8)
    # Model
    model = T5ModelForLoss(model_config)
    mim_trainer = Trainer(task='translation',
                          model=model,
                          config=config,
                          train_dataset=dataset)
    mim_trainer.train(resume_or_finetune_from_checkpoint=False)
    shutil.rmtree(dir_path, ignore_errors=True)
