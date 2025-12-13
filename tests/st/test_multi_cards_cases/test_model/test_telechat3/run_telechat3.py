# Copyright 2025 TeleAI Technologies Co., Ltd
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
Test module for testing the paralleled mcore telechat3 interface used for mindformers.
"""
import os
import argparse
from types import MethodType

import mindspore as ms
from tests.st.training_checker import TrainingChecker
from data_gen_utils import get_dataset, generate_weight

from mindformers import build_context, MindFormerConfig
from mindformers.trainer import Trainer

CUR_DIR = os.path.dirname(__file__)

ms.set_context(mode=ms.GRAPH_MODE)


def telechat3_train(config, dataset, construct_args_key, checker_config):
    """set model train."""
    callback = TrainingChecker(**checker_config)
    task_trainer = Trainer(task='text_generation',
                           args=config,
                           train_dataset=dataset,
                           callbacks=callback)

    task_trainer.config.train_dataset.input_columns = construct_args_key
    task_trainer.config.train_dataset.construct_args_key = construct_args_key
    def create_network(self, default_args):
        network = type(self).create_network(self, default_args)
        param_dict = generate_weight(network)
        ms.load_param_into_net(network, param_dict)
        return network
    task_trainer.trainer.create_network = MethodType(create_network, task_trainer.trainer)
    task_trainer.train()
    if checker_config.get('experiment_mode'):
        callback.get_experiment_results()


def parallel_train_dp2_mp2_cp2_ep2():
    """test mcore telechat3 train in dp=mp=ep=2."""
    ms.set_seed(0)
    config = MindFormerConfig(f'{CUR_DIR}/telechat3_train.yaml')
    config.print_separate_loss = False
    config.train_precision_sync = True
    config.pretrained_model_dir = CUR_DIR
    config.parallel.full_batch = False
    config.parallel.dataset_strategy = 'full_batch'
    config.parallel_config.context_parallel = 2
    config.parallel_config.pipeline_stage = 1
    build_context(config)

    construct_args_key = ['input_ids', 'labels']
    model_config = config.model.model_config
    dataset = get_dataset(model_config.seq_length, model_config.vocab_size, 2, 20)

    loss_std = [15.319313, 15.319500, 15.319048, 15.318926, 15.319642,
                15.319432, 15.319224, 15.319257, 15.319370, 15.319226,
                15.319025, 15.318974, 15.318835, 15.318989, 15.319075,
                15.318857, 15.318972, 15.319574, 15.319300, 15.319372,
                ]
    checker_config = {
        'loss_list_std': loss_std,
        'experiment_mode': False,
        'micro_batch_num': 1,
        'micro_batch_interleave_num': 1
    }
    telechat3_train(config, dataset, construct_args_key, checker_config)


TEST_MAP = {
    'parallel_train_dp2_mp2_cp2_ep2': parallel_train_dp2_mp2_cp2_ep2,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of telechat model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
