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
Test module for testing the paralleled deepseek3 interface used for mindformers.
How to run this:
    pytest tests/st/test_model/test_deepseek3_model/test_parallel.py
"""
import json
import argparse

import mindspore as ms
from mindformers import build_context
from mindformers.version_control import set_ms_deterministic

from tests.utils.model_tester import ModelTester
from base_model import get_config, get_model

ms.set_context(mode=ms.GRAPH_MODE)


def parallel_train_gmm_dp2mp2ep4pp2():
    """test deepseekv3 train using GroupedMatmul."""
    # dp=2, mp=2, pp=2, ep=4, micro_batch_num=2
    parallel_config = {
        'use_parallel': True,
        'data_parallel': 2,
        'model_parallel': 2,
        'expert_parallel': 4,
        'pipeline_stage': 2,
        'micro_batch_num': 2,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True,
        'vocab_emb_dp': True
    }

    # set model runner
    runner = ModelTester(run_mode='train', batch_size=4, step_num=10, experiment_mode=False, **parallel_config)
    build_context(runner.args)

    log_path = "log_train_gmm_dp2mp2ep4pp2"

    # set speed up json
    parallel_speed_up_json = {
        "matmul_grad_comm_overlap": True
    }

    # use json module;
    with open(f'{log_path}/parallel_speed_up.json', 'w') as f:
        json.dump(parallel_speed_up_json, f, indent=4)

    # get model config.
    model_config = get_config()
    model_config.parallel_config = runner.args.get_parallel_config()
    model = get_model(model_config)

    # set golden loss
    loss_std = [12.255825, 12.256344, 12.255493, 12.257595, 12.255733,
                12.257393, 12.254940, 12.257925, 12.256860, 12.258802]

    # set golden step time
    # self-test result: 181ms
    time_std = 200

    checker_config = {
        'micro_batch_num': 2,
    }
    set_ms_deterministic(True)
    ms.set_context(ascend_config={"parallel_speed_up_json_path": f"./{log_path}/parallel_speed_up.json"})
    runner.set_train(model, model_config, loss_std=loss_std, avg_time_std=time_std,
                     checker_config=checker_config)


def parallel_train_gmm_dp2mp2ep2pp2_deredundency():
    """test deepseekv3 train using deredundency"""
    # dp=2, mp=2, pp=2, ep=2
    parallel_config = {
        'use_parallel': True,
        'data_parallel': 2,
        'model_parallel': 2,
        'expert_parallel': 2,
        'pipeline_stage': 2,
        'micro_batch_num': 2,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True,
        'vocab_emb_dp': True,
    }

    # set model runner
    runner = ModelTester(run_mode='train', batch_size=4, step_num=10, experiment_mode=False, **parallel_config)
    build_context(runner.args)

    log_path = "log_train_gmm_dp2mp2ep2pp2_deredundency"

    # set speed up json
    parallel_speed_up_json = {
        "matmul_grad_comm_overlap": True
    }

    # use json module;
    with open(f'{log_path}/parallel_speed_up.json', 'w') as f:
        json.dump(parallel_speed_up_json, f, indent=4)

    # set the parameters to replace the base config.
    test_config = {'enable_deredundency': True,
                   'npu_nums_per_device': 2}

    model_config = get_config(test_config)
    model_config.parallel_config = runner.args.get_parallel_config()
    model = get_model(model_config)

    # set golden loss
    loss_std = [12.256952, 12.257590, 12.257160, 12.256790, 12.255981,
                12.257961, 12.255569, 12.259453, 12.257383, 12.258861]

    # set golden step time
    # CI result: 298ms
    time_std = 330

    checker_config = {
        'micro_batch_num': 2,
    }

    set_ms_deterministic(True)
    ms.set_context(ascend_config={"parallel_speed_up_json_path": f"./{log_path}/parallel_speed_up.json"})
    runner.set_train(model, model_config, loss_std=loss_std, avg_time_std=time_std,
                     checker_config=checker_config)


def parallel_train_bmm_dp2mp2ep2pp2():
    """test deepseekv3 train using BatchMatmul."""
    # dp=2, mp=2, pp=2, micro_batch_num=2
    parallel_config = {
        'use_parallel': True,
        'data_parallel': 2,
        'model_parallel': 2,
        'expert_parallel': 2,
        'pipeline_stage': 2,
        'micro_batch_num': 2,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True,
        'vocab_emb_dp': True,
    }
    runner = ModelTester(run_mode='train', batch_size=4, step_num=10, experiment_mode=False, **parallel_config)
    build_context(runner.args)

    log_path = "log_train_bmm_dp2mp2ep2pp2"
    parallel_speed_up_json = {
        "matmul_grad_comm_overlap": True
    }

    # use json module;
    with open(f'{log_path}/parallel_speed_up.json', 'w') as f:
        json.dump(parallel_speed_up_json, f, indent=4)

    # set the parameters to replace the base config.
    test_config = {'use_gmm': False}

    model_config = get_config(test_config)
    model_config.parallel_config = runner.args.get_parallel_config()
    model = get_model(model_config)

    # set golden loss
    loss_std = [12.258962, 12.256533, 12.257041, 12.252623, 12.253700,
                12.258326, 12.259855, 12.255697, 12.259889, 12.257683]

    # set golden step time
    # self-test result: 162ms
    time_std = 180

    checker_config = {
        'micro_batch_num': 2,
    }

    set_ms_deterministic(True)
    ms.set_context(ascend_config={"parallel_speed_up_json_path": f"./{log_path}/parallel_speed_up.json"})
    runner.set_train(model, model_config, loss_std=loss_std, avg_time_std=time_std,
                     checker_config=checker_config)


TEST_MAP = {
    'parallel_train_gmm_dp2mp2ep4pp2': parallel_train_gmm_dp2mp2ep4pp2,
    'parallel_train_gmm_dp2mp2ep2pp2_deredundency': parallel_train_gmm_dp2mp2ep2pp2_deredundency,
    'parallel_train_bmm_dp2mp2ep2pp2': parallel_train_bmm_dp2mp2ep2pp2
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of deepseekv3 model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
