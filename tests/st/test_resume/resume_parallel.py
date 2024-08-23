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
Test module for testing resume training from specified checkpoint.
How to run this:
pytest tests/st/test_resume/test_parallel_resume.py
"""
import os
import json
import numpy as np

from mindspore.dataset import GeneratorDataset

from mindformers import build_context
from mindformers.tools.utils import (
    LOCAL_DEFAULT_PATH,
    get_real_rank,
    get_real_group_size
)
from mindformers.trainer import Trainer, TrainingArguments
from mindformers.models.gpt2 import GPT2LMHeadModel, GPT2Config


def generator():
    """dataset generator"""
    np.random.seed(42)
    seq_len = 1025
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    input_mask = np.ones_like(input_ids)
    train_data = (input_ids, input_mask)
    for _ in range(32):
        yield train_data


def gpt_trainer_train_from_instance():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """
    # Config definition
    config = TrainingArguments(
        num_train_epochs=2,
        batch_size=4,
        use_parallel=True,
        data_parallel=2,
        model_parallel=2,
        pipeline_stage=2,
        micro_batch_num=2,
        save_steps=1,
        save_directory=os.path.join(LOCAL_DEFAULT_PATH, "test_resume_parallel"),
    )

    build_context(config)

    # Model
    model_config = GPT2Config(num_layers=2)
    model = GPT2LMHeadModel(model_config)

    # Dataset and operations
    dataset = GeneratorDataset(generator, column_names=["input_ids", "input_mask"])
    dataset = dataset.batch(batch_size=8)

    trainer = Trainer(model=model,
                      args=config,
                      train_dataset=dataset,
                      reset_model=True)
    trainer.train(train_checkpoint=False)

    # wait other rank saving over
    meta_json = os.path.join(LOCAL_DEFAULT_PATH, "test_resume_parallel", "checkpoint",
                             "rank_{}".format(get_real_rank()), "meta.json")
    with open(meta_json, "r") as json_file:
        meta_data = json.load(json_file)
    last_epoch = meta_data["last_epoch"]
    last_step = meta_data["last_step"]
    while True:
        saving_over = True
        for rank_id_tmp in range(get_real_group_size()):
            meta_json = os.path.join(LOCAL_DEFAULT_PATH, "test_resume_parallel", "checkpoint",
                                     "rank_{}".format(rank_id_tmp), "meta.json")
            with open(meta_json, "r") as json_file:
                meta_data = json.load(json_file)
            compare_epoch = meta_data["last_epoch"]
            compare_step = meta_data["last_step"]
            if last_epoch != compare_epoch or last_step != compare_step:
                saving_over = False
        if saving_over:
            break

    checkpoint_dir = os.path.join(LOCAL_DEFAULT_PATH, "test_resume_parallel", "checkpoint",
                                  "rank_{}".format(get_real_rank()))
    output_checkpoint_path = [
        checkpoint for checkpoint in os.listdir(checkpoint_dir)
        if checkpoint.endswith('.ckpt')
    ]
    output_checkpoint_path = sorted(output_checkpoint_path,
                                    key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))

    for _ in range(2):
        os.remove(os.path.join(checkpoint_dir, output_checkpoint_path.pop()))

    # Dataset and operations
    dataset = GeneratorDataset(generator, column_names=["input_ids", "input_mask"])
    dataset = dataset.batch(batch_size=8)

    trainer = Trainer(model=model,
                      args=config,
                      train_dataset=dataset,
                      reset_model=True)
    trainer.train(resume_from_checkpoint=os.path.join(LOCAL_DEFAULT_PATH, "test_resume_parallel", "checkpoint"),
                  resume_training=True)

    # Dataset and operations
    dataset = GeneratorDataset(generator, column_names=["input_ids", "input_mask"])
    dataset = dataset.batch(batch_size=8)

    trainer = Trainer(model=model,
                      args=config,
                      train_dataset=dataset,
                      reset_model=True)
    trainer.train(resume_from_checkpoint=os.path.join(LOCAL_DEFAULT_PATH, "test_resume_parallel", "checkpoint"),
                  resume_training=output_checkpoint_path[-1])

gpt_trainer_train_from_instance()
