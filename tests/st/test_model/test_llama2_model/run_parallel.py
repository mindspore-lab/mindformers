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
Test module for testing the paralleled llama2 interface used for mindformers.
How to run this:
    pytest tests/st/test_model/test_llama2_model/test_parallel.py
"""
import argparse
import copy

import mindspore as ms
from mindformers import build_context
from mindformers.modules.transformer.transformer import TransformerOpParallelConfig
from mindformers.models.llama import LlamaConfig

from tests.utils.model_tester import ModelTester
from base_model import get_config, get_model, BASE_CONFIG

ms.set_context(mode=ms.GRAPH_MODE)


def parallel_train_mp2_pp2():
    """test llama2 train in model_parallel=2, pipeline_stage=2."""
    # dp=1, mp=2, pp=2, micro_batch_num=4, micro_batch_interleave_num=2
    parallel_config = {
        'use_parallel': True,
        'data_parallel': 1,
        'model_parallel': 2,
        'pipeline_stage': 2,
        'micro_batch_num': 4,
        'micro_batch_interleave_num': 2,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True,
        'vocab_emb_dp': False  # if set True, cause error
    }
    runner = ModelTester(run_mode='train', batch_size=8, experiment_mode=False, **parallel_config)
    build_context(runner.args)

    model_config = get_config()
    model_config.parallel_config = runner.args.get_parallel_config()
    model = get_model(model_config)

    loss_std = [10.631486, 10.623098, 10.622696, 10.614234, 10.608653,
                10.609907, 10.612187, 10.596552, 10.595687, 10.589638,
                10.586476, 10.587587, 10.577414, 10.573078, 10.577648,
                10.573657, 10.572372, 10.566246, 10.575710, 10.578465]
    time_std = 550

    checker_config = {
        'micro_batch_num': 4,
        'micro_batch_interleave_num': 2
    }
    runner.set_train(model, model_config, loss_std=loss_std, avg_time_std=time_std,
                     checker_config=checker_config)


def parallel_train_dp2():
    """test llama2 train in data_parallel=2."""
    # dp=2, parallel_optimizer
    parallel_config = {
        'use_parallel': True,
        'data_parallel': 2,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True
    }
    runner = ModelTester(run_mode='train', batch_size=2, experiment_mode=False, **parallel_config)
    build_context(runner.args)

    model_config = get_config()
    model_config.parallel_config = runner.args.get_parallel_config()
    model = get_model(model_config)

    loss_std = [10.629690, 10.622313, 10.627311, 10.609977, 10.618529,
                10.624242, 10.613264, 10.621710, 10.622666, 10.614981,
                10.605037, 10.601599, 10.591890, 10.595872, 10.601541,
                10.573302, 10.586630, 10.598871, 10.602546, 10.578698]
    time_std = 280

    runner.set_train(model, model_config, loss_std=loss_std, avg_time_std=time_std)


def parallel_train_mp2_cp2():
    """test llama2 train in model_parallel=2, context_parallel=2."""
    # dp=2, parallel_optimizer
    parallel_config = {
        'use_parallel': True,
        'use_seq_parallel': False,
        'enable_parallel_optimizer': True
    }
    runner = ModelTester(run_mode='train', batch_size=2, experiment_mode=False, **parallel_config)
    build_context(runner.args)

    model_config = get_config()
    # only support mp with cp
    model_config.parallel_config = TransformerOpParallelConfig(model_parallel=2,
                                                               context_parallel=2,
                                                               use_seq_parallel=False)
    model = get_model(model_config)

    loss_std = [10.627013, 10.620491, 10.614843, 10.616434, 10.625935,
                10.623493, 10.600618, 10.629920, 10.611515, 10.610081,
                10.599187, 10.607603, 10.608433, 10.595035, 10.598252,
                10.594250, 10.595692, 10.598722, 10.586823, 10.604011]
    time_std = 400
    context_parallel = {
        'data_parallel': 1,
        'model_parallel': 2,
        'context_parallel': 2,
        'pipeline_stage': 1,
    }
    runner.set_train(model, model_config, loss_std=loss_std, avg_time_std=time_std,
                     parallel_config=context_parallel)


def parallel_train_sapp_mp2_pp2():
    """test llama2 train in auto_parallel and model_parallel=2, pipeline_stage=2."""
    # dp=2, parallel_optimizer
    parallel_config = {
        'use_parallel': True,
        'data_parallel': 1,
        'model_parallel': 2,
        'pipeline_stage': 2,
        'micro_batch_num': 4,
        'micro_batch_interleave_num': 2,
        'enable_parallel_optimizer': True,
        'parallel_mode': 2,  # ms.ParallelMode.AUTO_PARALLEL
        'search_mode': "recursive_programming"
    }
    runner = ModelTester(run_mode='train', batch_size=8, experiment_mode=False, **parallel_config)
    build_context(runner.args)

    model_config = get_config()
    model_config.parallel_config = runner.args.get_parallel_config()
    model = get_model(model_config)

    loss_std = [10.631486, 10.623105, 10.622698, 10.614237, 10.608654,
                10.609909, 10.612190, 10.596548, 10.595697, 10.589638,
                10.586469, 10.587587, 10.577406, 10.573081, 10.577652,
                10.573650, 10.572378, 10.566244, 10.575705, 10.578461]
    time_std = 600

    checker_config = {
        'micro_batch_num': 4,
        'micro_batch_interleave_num': 2
    }
    runner.set_train(model, model_config, loss_std=loss_std, avg_time_std=time_std,
                     checker_config=checker_config)


def train_input_sliced():
    """test llama2 train when input has been processed to seq_length."""
    # dp=2, parallel_optimizer
    parallel_config = {
        'use_parallel': True,
        'data_parallel': 2,
        'model_parallel': 1,
        'pipeline_stage': 1,
        'micro_batch_num': 1,
    }
    runner = ModelTester(run_mode='train', batch_size=8, input_sliced_sig=True, use_label=True, **parallel_config)
    build_context(runner.args)

    model_config = get_config()
    model_config.input_sliced_sig = True
    model_config.parallel_config = runner.args.get_parallel_config()
    model = get_model(model_config)

    loss_std = [10.215071, 10.156379, 10.108549, 10.063042, 10.014468,
                9.968597, 9.926680, 9.887671, 9.854958, 9.827715,
                9.800526, 9.775367, 9.748009, 9.740524, 9.717923,
                9.719361, 9.709933, 9.718268, 9.714646, 9.706030]
    time_std = 700

    runner.set_train(model, model_config, loss_std=loss_std, avg_time_std=time_std)


def parallel_train_ndtp_cp2x2y2z1():
    """test llama2 train in context_parallel=2, model_parallel=4, use_3d_tensor_parallel =True,
    tp_x=2, tp_y=2, tp_z=1"""
    parallel_config = {
        'use_parallel': True,
        'use_seq_parallel': True,
    }
    runner = ModelTester(run_mode='train', batch_size=8, experiment_mode=False, **parallel_config)
    build_context(runner.args)
    base_config = copy.deepcopy(BASE_CONFIG)
    base_config['use_3d_tensor_parallel'] = True
    base_config['tp_x'] = 2
    base_config['tp_y'] = 2
    base_config['tp_z'] = 1
    model_config = LlamaConfig(**base_config)
    model_config.parallel_config = runner.args.get_parallel_config()
    model = get_model(model_config)
    context_parallel = {
        'data_parallel': 1,
        'model_parallel': 4,
        'context_parallel': 2,
        'pipeline_stage': 1,
        'use_seq_parallel': True,
    }

    loss_std = [10.630941, 10.622524, 10.613161, 10.604401, 10.592041,
                10.577415, 10.578622, 10.567156, 10.556034, 10.547044,
                10.545972, 10.542600, 10.537585, 10.526172, 10.522586,
                10.522101, 10.521068, 10.521736, 10.522684, 10.521909]
    time_std = 1170
    runner.set_train(model, model_config, loss_std=loss_std, avg_time_std=time_std, parallel_config=context_parallel)


TEST_MAP = {
    'parallel_train_mp2_pp2': parallel_train_mp2_pp2,
    'parallel_train_dp2': parallel_train_dp2,
    'parallel_train_mp2_cp2': parallel_train_mp2_cp2,
    'parallel_train_sapp_mp2_pp2': parallel_train_sapp_mp2_pp2,
    "train_input_sliced": train_input_sliced,
    'parallel_train_ndtp_cp2x2y2z1': parallel_train_ndtp_cp2x2y2z1
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of llama2 model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
