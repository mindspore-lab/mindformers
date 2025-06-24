# Copyright 2025 Huawei Technologies Co., Ltd
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
Test module for testing the paralleled mcore deepseek3 interface used for mindformers.
"""
import os
import argparse

import mindspore as ms

from mindformers import build_context, MindFormerConfig
from mindformers.trainer import Trainer

from tests.st.training_checker import TrainingChecker

from data_gen_utils import get_tnd_dataset, get_dataset

CUR_DIR = os.path.dirname(__file__)

ms.set_context(mode=ms.GRAPH_MODE)

def ds3_train(config, dataset, construct_args_key, checker_config):
    """set model train."""
    callback = TrainingChecker(**checker_config)
    task_trainer = Trainer(task="text_generation",
                           args=config,
                           train_dataset=dataset,
                           callbacks=callback)

    task_trainer.config.train_dataset.construct_args_key = construct_args_key

    task_trainer.train()
    if checker_config.get('experiment_mode'):
        callback.get_experiment_results()

def parallel_train_dp2_mp2_ep2():
    """test mcore deepseekv3 train in dp=mp=ep=2."""
    ms.set_seed(0)
    config = MindFormerConfig(f'{CUR_DIR}/deepseekv3_train.yaml')
    config.train_precision_sync = True
    config.pretrained_model_dir = CUR_DIR
    config.runner_config.sink_mode = False
    config.parallel.full_batch = True
    config.parallel.dataset_strategy = 'full_batch'
    config.parallel_config.pipeline_stage = 1
    build_context(config)

    construct_args_key = ['input_ids', 'labels']
    model_config = config.model.model_config
    dataset = get_dataset(model_config.seq_length, model_config.vocab_size, 4, 20)

    loss_std = [13.555495, 13.551261, 13.551877, 13.552636, 13.556355,
                13.551405, 13.552004, 13.551588, 13.551432, 13.548488,
                13.547302, 13.551485, 13.551678, 13.552629, 13.550841,
                13.546679, 13.545593, 13.554431, 13.549174, 13.543098]
    time_std = 480
    checker_config = {
        'loss_list_std': loss_std,
        'avg_step_time_std': time_std,
        'experiment_mode': False,
        'micro_batch_num': 1,
        'micro_batch_interleave_num': 1
    }
    ds3_train(config, dataset, construct_args_key, checker_config)

def parallel_train_dp2_pp2_ep2_tnd():
    """test mcore deepseekv3 train in dp=pp=ep=2 with TND layout."""
    config = MindFormerConfig(f'{CUR_DIR}/deepseekv3_train.yaml')
    config.train_precision_sync = True
    config.pretrained_model_dir = CUR_DIR
    config.runner_config.sink_mode = True
    config.parallel.full_batch = False
    dp = config.parallel_config.data_parallel
    config.parallel_config.model_parallel = 1
    config.parallel_config.use_seq_parallel = False
    config.parallel.dataset_strategy = [[dp, 1], [dp, 1], [dp, 1]]
    build_context(config)

    construct_args_key = ['input_ids', 'labels', 'actual_seq_len']
    model_config = config.model.model_config
    dataset = get_tnd_dataset(
        model_config.seq_length,
        model_config.vocab_size,
        config.parallel_config.micro_batch_num,
        batch_size=4, step_num=20
    )

    loss_std = [13.549505, 13.548901, 13.554827, 13.550955, 13.550215,
                13.560290, 13.554025, 13.556352, 13.549892, 13.551298,
                13.547314, 13.552792, 13.549354, 13.550882, 13.553068,
                13.549754, 13.550379, 13.556471, 13.551868, 13.548780]
    checker_config = {
        'loss_list_std': loss_std,
        'avg_step_time_std': None, # do not monitor performance because of fluctuations in tnd
        'experiment_mode': False,
        'micro_batch_num': 2,
        'micro_batch_interleave_num': 1
    }
    ds3_train(config, dataset, construct_args_key, checker_config)

TEST_MAP = {
    'parallel_train_dp2_mp2_ep2': parallel_train_dp2_mp2_ep2,
    'parallel_train_dp2_pp2_ep2_tnd': parallel_train_dp2_pp2_ep2_tnd,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of deepseek model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
