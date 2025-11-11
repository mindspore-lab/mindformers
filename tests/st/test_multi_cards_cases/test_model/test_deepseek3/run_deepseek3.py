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
from types import MethodType

import mindspore as ms
from tests.st.training_checker import TrainingChecker
from data_gen_utils import get_tnd_dataset, get_dataset, generate_weight

from mindformers import build_context, MindFormerConfig
from mindformers.trainer import Trainer

CUR_DIR = os.path.dirname(__file__)

ms.set_context(mode=ms.GRAPH_MODE)


def ds3_train(config, dataset, construct_args_key, checker_config):
    """set model train."""
    callback = TrainingChecker(**checker_config)
    task_trainer = Trainer(task="text_generation",
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
    """test mcore deepseekv3 train in dp=mp=ep=2."""
    ms.set_seed(0)
    config = MindFormerConfig(f'{CUR_DIR}/deepseekv3_train.yaml')
    config.print_separate_loss = False
    config.train_precision_sync = True
    config.pretrained_model_dir = CUR_DIR
    config.parallel.full_batch = True
    config.parallel.dataset_strategy = 'full_batch'
    config.parallel_config.context_parallel = 2
    config.parallel_config.pipeline_stage = 1
    build_context(config)

    construct_args_key = ['input_ids', 'labels']
    model_config = config.model.model_config
    dataset = get_dataset(model_config.seq_length, model_config.vocab_size, 2, 20)

    loss_std = [13.485943, 13.485964, 13.485873, 13.486125, 13.486035,
                13.486079, 13.485928, 13.485991, 13.485875, 13.485889,
                13.485916, 13.486242, 13.485828, 13.485985, 13.486170,
                13.485849, 13.486152, 13.485961, 13.486010, 13.486052,
                ]
    checker_config = {
        'loss_list_std': loss_std,
        'experiment_mode': False,
        'micro_batch_num': 1,
        'micro_batch_interleave_num': 1
    }
    ds3_train(config, dataset, construct_args_key, checker_config)


def parallel_train_dp2_pp2_ep2_tnd():
    """test mcore deepseekv3 train in dp=pp=ep=2 with TND layout."""
    ms.set_seed(0)
    config = MindFormerConfig(f'{CUR_DIR}/deepseekv3_train.yaml')
    config.print_separate_loss = False
    config.model.model_config.use_eod_attn_mask_compression = True
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

    loss_std = [13.485911, 13.485952, 13.486026, 13.485985, 13.485933,
                13.486032, 13.485978, 13.485965, 13.486005, 13.485919,
                13.485975, 13.485991, 13.485999, 13.485950, 13.485901,
                13.486069, 13.485883, 13.485929, 13.485968, 13.485967,]
    checker_config = {
        'loss_list_std': loss_std,
        'experiment_mode': False,
        'micro_batch_num': 2,
        'micro_batch_interleave_num': 1
    }
    ds3_train(config, dataset, construct_args_key, checker_config)


def parallel_train_dp2_mp2_ep2_calculate_per_token_loss_and_print_seperate_loss():
    """test mcore deepseekv3 train in dp=mp=ep=2."""
    ms.set_seed(0)
    config = MindFormerConfig(f'{CUR_DIR}/deepseekv3_train.yaml')
    config.print_separate_loss = True
    config.calculate_per_token_loss = True

    config.train_precision_sync = True
    config.pretrained_model_dir = CUR_DIR
    config.parallel.full_batch = True
    config.parallel.dataset_strategy = 'full_batch'
    config.parallel_config.pipeline_stage = 1
    build_context(config)

    construct_args_key = ['input_ids', 'labels']
    model_config = config.model.model_config
    dataset = get_dataset(model_config.seq_length, model_config.vocab_size, 4, 20)

    loss_std = [13.485077, 13.485191, 13.485246, 13.485250, 13.485148,
                13.485259, 13.485207, 13.485249, 13.485296, 13.485240,
                13.485134, 13.485152, 13.485264, 13.485212, 13.485147,
                13.485227, 13.485124, 13.485272, 13.485239, 13.485266,]

    checker_config = {
        'loss_list_std': loss_std,
        'experiment_mode': False,
        'micro_batch_num': 1,
        'micro_batch_interleave_num': 1
    }
    ds3_train(config, dataset, construct_args_key, checker_config)


def moe_token_permute():
    """test mcore deepseekv3 train in moe_token_permute=True."""
    ms.set_seed(0)
    config = MindFormerConfig(f'{CUR_DIR}/deepseekv3_train.yaml')
    config.print_separate_loss = False
    config.train_precision_sync = True
    config.pretrained_model_dir = CUR_DIR
    config.parallel.full_batch = True
    config.parallel.dataset_strategy = 'full_batch'
    config.model.model_config.moe_permute_fusion = True
    build_context(config)

    construct_args_key = ['input_ids', 'labels']
    model_config = config.model.model_config
    dataset = get_dataset(model_config.seq_length, model_config.vocab_size, 4, 20)

    loss_std = [13.485840, 13.485862, 13.486006, 13.486031, 13.485930,
                13.485970, 13.485985, 13.485969, 13.486046, 13.485975,
                13.485966, 13.486019, 13.485966, 13.485914, 13.485946,
                13.486049, 13.485964, 13.485968, 13.485966, 13.486050,]
    checker_config = {
        'loss_list_std': loss_std,
        'experiment_mode': False,
        'micro_batch_num': 2,
        'micro_batch_interleave_num': 1
    }
    ds3_train(config, dataset, construct_args_key, checker_config)

def parallel_train_pp2_mp2_ep2_zbv():
    """test mcore deepseekv3 train in pp=mp=ep=2 with zero_bubble_v."""
    ms.set_seed(0)
    config = MindFormerConfig(f'{CUR_DIR}/deepseekv3_train.yaml')
    config.print_separate_loss = False
    config.train_precision_sync = True
    config.pretrained_model_dir = CUR_DIR
    config.runner_config.sink_mode = True
    config.parallel.full_batch = True
    config.parallel.dataset_strategy = 'full_batch'
    config.parallel_config.data_parallel = 1
    config.parallel_config.micro_batch_num = 4
    config.parallel.pipeline_config = {'pipeline_interleave': True, 'pipeline_scheduler': 'zero_bubble_v'}
    config.model.model_config.pp_interleave_num = 2

    build_context(config)

    construct_args_key = ['input_ids', 'labels']
    model_config = config.model.model_config
    dataset = get_dataset(model_config.seq_length, model_config.vocab_size, 4, 20)

    loss_std = [13.485840, 13.485862, 13.486006, 13.486031, 13.485930,
                13.485970, 13.485985, 13.485970, 13.486046, 13.485975,
                13.485965, 13.486017, 13.485967, 13.485915, 13.485946,
                13.486049, 13.485963, 13.485968, 13.485966, 13.486050,]
    checker_config = {
        'loss_list_std': loss_std,
        'experiment_mode': False,
        'micro_batch_num': 4,
        'micro_batch_interleave_num': 1,
        'zero_bubble_v': True
    }
    ds3_train(config, dataset, construct_args_key, checker_config)

def moe_eplb():
    """test mcore deepseekv3 train in moe_eplb=True."""
    ms.set_seed(0)
    config = MindFormerConfig(f'{CUR_DIR}/deepseekv3_train.yaml')
    config.print_separate_loss = False
    config.train_precision_sync = True
    config.pretrained_model_dir = CUR_DIR
    config.parallel.full_batch = True
    config.parallel.dataset_strategy = 'full_batch'
    config.model.model_config.print_expert_load = True
    build_context(config)

    construct_args_key = ['input_ids', 'labels']
    model_config = config.model.model_config
    dataset = get_dataset(model_config.seq_length, model_config.vocab_size, 4, 20)

    loss_std = [13.485840, 13.485862, 13.486006, 13.486031, 13.485930,
                13.485970, 13.485985, 13.485969, 13.486046, 13.485975,
                13.485966, 13.486019, 13.485966, 13.485914, 13.485946,
                13.486049, 13.485964, 13.485968, 13.485966, 13.486050,]
    checker_config = {
        'loss_list_std': loss_std,
        'experiment_mode': False,
        'micro_batch_num': 2,
        'micro_batch_interleave_num': 1
    }
    ds3_train(config, dataset, construct_args_key, checker_config)


TEST_MAP = {
    'parallel_train_dp2_pp2_ep2_tnd': parallel_train_dp2_pp2_ep2_tnd,
    "parallel_train_dp2_mp2_ep2_calculate_per_token_loss_and_print_seperate_loss":
        parallel_train_dp2_mp2_ep2_calculate_per_token_loss_and_print_seperate_loss,
    'parallel_train_pp2_mp2_ep2_zbv': parallel_train_pp2_mp2_ep2_zbv,
    "moe_token_permute": moe_token_permute,
    "moe_eplb": moe_eplb,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of deepseek model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
