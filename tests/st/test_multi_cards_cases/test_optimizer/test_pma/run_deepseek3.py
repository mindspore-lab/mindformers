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
from safetensors import safe_open
import numpy as np

import mindspore as ms
from mindspore.ops.operations import Cast

from tests.st.test_multi_cards_cases.test_model.test_deepseek3.data_gen_utils import get_dataset, generate_weight

from mindformers import build_context, MindFormerConfig
from mindformers.trainer import Trainer
from mindformers.core.callback.callback import CheckpointMonitor, TrainCallBack


cpu_cast = Cast().set_device("CPU")

CUR_DIR = os.path.dirname(__file__)

ms.set_context(mode=ms.GRAPH_MODE)

OPTIMIZER_KEYS = {"adam_m", "adam_v", "epoch_num", "global_step",
                  "loss_scale", "step_num", "scale_sense"}


def ds3_train(config, dataset, construct_args_key, callback):
    """set model train."""
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


def create_config(mode='origin'):
    """Create config for different test modes."""
    ms.set_seed(0)
    config = MindFormerConfig(f'{CUR_DIR}/deepseekv3_train.yaml')
    config.print_separate_loss = False
    config.train_precision_sync = True
    config.pretrained_model_dir = CUR_DIR
    config.parallel.full_batch = True
    config.parallel.dataset_strategy = 'full_batch'
    config.parallel_config.data_parallel = 1

    if mode != 'load_origin':
        config.load_checkpoint = f'{CUR_DIR}/test/checkpoint/'

    if mode == 'ema':
        config.optimizer.type = 'PmaAdamW'
        config.optimizer.interleave_step = 1
        config.optimizer.fused_num = 2
    elif mode == 'sma':
        config.optimizer.type = 'PmaAdamW'
        config.optimizer.interleave_step = 1
        config.optimizer.fused_num = 2
        config.optimizer.fused_algo = 'sma'

    build_context(config)
    return config


def run_load_origin():
    """Run origin test for loading checkpoint."""
    ms.set_seed(0)
    config = create_config('load_origin')

    construct_args_key = ['input_ids', 'labels']
    model_config = config.model.model_config
    dataset = get_dataset(model_config.seq_length, model_config.vocab_size, 4, 20)

    callback = [CheckpointMonitor(
        save_checkpoint_steps=1,
        checkpoint_format='safetensors',
        directory=f"{CUR_DIR}/test"),
                TrainCallBack(stop_step=1)]

    ds3_train(config, dataset, construct_args_key, callback)


def run_origin():
    """Run origin test to compare."""
    ms.set_seed(0)
    config = create_config('origin')

    construct_args_key = ['input_ids', 'labels']
    model_config = config.model.model_config
    dataset = get_dataset(model_config.seq_length, model_config.vocab_size, 4, 20)

    callback = [
        CheckpointMonitor(save_checkpoint_steps=1, checkpoint_format='safetensors', directory=f"{CUR_DIR}/origin"),
        TrainCallBack(stop_step=2)]
    ds3_train(config, dataset, construct_args_key, callback)


def run_ema():
    """Run ema test to compare."""
    ms.set_seed(0)
    config = create_config('ema')

    construct_args_key = ['input_ids', 'labels']
    model_config = config.model.model_config
    dataset = get_dataset(model_config.seq_length, model_config.vocab_size, 4, 20)

    callback = [CheckpointMonitor(save_checkpoint_steps=1, checkpoint_format='safetensors', directory=f"{CUR_DIR}/ema"),
                TrainCallBack(stop_step=2)]
    ds3_train(config, dataset, construct_args_key, callback)


def run_sma():
    """Run sma test to compare."""
    ms.set_seed(0)
    config = create_config('sma')

    construct_args_key = ['input_ids', 'labels']
    model_config = config.model.model_config
    dataset = get_dataset(model_config.seq_length, model_config.vocab_size, 4, 20)

    callback = [CheckpointMonitor(save_checkpoint_steps=1, checkpoint_format='safetensors', directory=f"{CUR_DIR}/sma"),
                TrainCallBack(stop_step=2)]
    ds3_train(config, dataset, construct_args_key, callback)


def test_pma():
    """
    Feature: test pma.
    Description: Run pma function.
    Expectation: Success or assert precision failed
    """
    run_load_origin()
    run_origin()
    run_ema()
    run_sma()
    origin_dict1 = load_safetensors("CKP_rank_0-1_1.safetensors", f"{CUR_DIR}/origin/checkpoint/rank_0")
    origin_dict2 = load_safetensors("CKP_rank_0-2_1.safetensors", f"{CUR_DIR}/origin/checkpoint/rank_0")

    check_dict1 = load_safetensors("CKP_rank_0-1_1.safetensors", f"{CUR_DIR}/ema/checkpoint/rank_0")
    check_dict2 = load_safetensors("CKP_rank_0-2_1.safetensors", f"{CUR_DIR}/ema/checkpoint/rank_0")

    check_dict3 = load_safetensors("CKP_rank_0-1_1.safetensors", f"{CUR_DIR}/sma/checkpoint/rank_0")
    check_dict4 = load_safetensors("CKP_rank_0-2_1.safetensors", f"{CUR_DIR}/sma/checkpoint/rank_0")

    compare_checkpoint_step_one(check_dict1, origin_dict1, 0.2, "pma_weight_ema.")
    compare_checkpoint_step_one(check_dict3, origin_dict1, 1, "pma_weight_sma.")

    compare_checkpoint_step_two(check_dict2, origin_dict1, origin_dict2, 0.2 * 0.8, 0.2)
    compare_checkpoint_step_two(check_dict4, origin_dict1, origin_dict2, 0.5, 0.5)

TEST_MAP = {
    "test_pma": test_pma,
}


def compare_checkpoint_step_one(check_dict, origin_dict, alpha, pma_prefix):
    """Compare checkpoint for first step."""
    unexpected_keys = []
    for k, _ in check_dict.items():
        if origin_dict.get(k) is not None:
            origin_value = cpu_cast(ms.Tensor(origin_dict.get(k)), ms.float32)
            check_value = cpu_cast(ms.Tensor(check_dict.get(k)), ms.float32)
            assert np.allclose(origin_value, check_value)
        else:
            if 'pma' in k:
                pma_value = cpu_cast(ms.Tensor(check_dict.get(k)), ms.float32)
                assert origin_dict.get(k.replace(pma_prefix, "")) is not None
                origin_value = cpu_cast(ms.Tensor(origin_dict.get(k.replace(pma_prefix, ""))),
                                        ms.float32)
                assert np.allclose(pma_value, alpha * origin_value)
            else:
                unexpected_keys.append(k)
    assert not unexpected_keys


def compare_checkpoint_step_two(check_dict, origin_dict1, origin_dict2, alpha1, alpha2):
    """Compare checkpoint for second step."""
    unexpected_keys = []
    for k, _ in check_dict.items():
        if "router.expert_bias" in k:
            continue
        if origin_dict2.get(k) is not None and origin_dict1.get(k) is not None:
            origin_value1 = cpu_cast(ms.Tensor(origin_dict1.get(k)), ms.float32)
            origin_value2 = cpu_cast(ms.Tensor(origin_dict2.get(k)), ms.float32)
            check_value = cpu_cast(ms.Tensor(check_dict.get(k)), ms.float32)
            if any(key in k for key in OPTIMIZER_KEYS):
                assert np.allclose(origin_value2, check_value)
                continue
            assert np.allclose(origin_value1 * alpha1 + alpha2 * origin_value2, check_value)
        else:
            if 'pma' in k:
                pma_value = cpu_cast(ms.Tensor(check_dict.get(k)), ms.float32)
                assert np.allclose(pma_value, pma_value * 0)
            else:
                unexpected_keys.append(k)
    assert not unexpected_keys


def load_safetensors(ckpt, path):
    """Load checkpoint which format is safetensors."""
    ckpt = os.path.join(path, ckpt)
    ckpt_dict = {}

    with safe_open(ckpt, framework='np', device='cpu') as f:
        for k in f.keys():
            ckpt_dict[k] = f.get_tensor(k)

    return ckpt_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of deepseek model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
