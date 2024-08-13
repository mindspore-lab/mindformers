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
"""Quant llama2 to w8a8."""

import os
import argparse
import time

import mindspore as ms
from mindspore import dtype as msdtype
from mindspore.communication import get_rank

from mindformers import LlamaForCausalLM, MindFormerConfig

from mindspore_gs.ptq import PTQMode, PTQConfig
from mindspore_gs.common import BackendTarget, logger
from mindspore_gs.datasets import get_datasets
from mindspore_gs.ptq.smooth_quant import SmoothQuant as SQ
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--dataset_type', '-t', type=str, required=True)
    parser.add_argument('--dataset_path', '-s', type=str, required=True)
    parser.add_argument('--load_checkpoint', '-l', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--world_size', '-w', type=int, required=True)
    args = parser.parse_args()
    args.dataset_type = args.dataset_type.lower()
    logger.info(f"quant args: {args}")
    return args


def quant_network(net: LlamaForCausalLM, net_helper, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, **kwargs):
    """Quant llama2 model to w8a16 with RTN algorithm."""
    start_time = time.time()
    logger.info("Use SmoothQuant algo to quant network and weight.")
    cfg = PTQConfig(mode=mode, backend=backend, opname_blacklist=["w2", "lm_head"], act_dtype=msdtype.int8)
    ptq = SQ(config=cfg)
    logger.info(f'Create PTQ cost time is {time.time() - start_time} s.')
    start_time = time.time()
    ds_path = kwargs.get("ds_path", "")
    if not ds_path:
        raise ValueError("Please provide datasets for calibrating.")
    ds_type = kwargs.get("ds_type", "")
    if not ds_type:
        raise ValueError("Please provide datasets type for calibrating.")
    bs_ = net_helper.get_spec('batch_size')
    seq_ = net_helper.get_spec('seq_length')
    max_decode_length = net_helper.get_spec('max_decode_length')
    ignore_token_id = net_helper.get_spec('ignore_token_id')
    tokenizer = net_helper.create_tokenizer()
    ds = get_datasets(ds_type, ds_path, "train", bs_, seq_, max_decode_length, tokenizer, ignore_token_id, 1, False,
                      n_samples=200)
    net = ptq.apply(net, net_helper, datasets=ds)
    logger.info(f'Apply PTQ cost time is {time.time() - start_time} s.')
    start_time = time.time()
    net.phase = "quant_convert"
    net = ptq.convert(net)
    logger.info(f'Convert to real quantize cost time is {time.time() - start_time} s.')
    return net


if __name__ == "__main__":
    start = time.time()
    uargs = get_args()
    logger.info('Creating network...')

    config = MindFormerConfig(uargs.config_path)
    if uargs.load_checkpoint:
        config.load_checkpoint = uargs.load_checkpoint
    if uargs.output_dir:
        config.output_dir = uargs.output_dir
    if uargs.world_size:
        if config.parallel_config.model_parallel != uargs.world_size:
            logger.info(
                f"world_size is {uargs.world_size}, not equal to \
                config.parallel_config.model_parallel:{config.parallel_config.model_parallel}")
            config.parallel_config.model_parallel = uargs.world_size
            logger.info(f"reset config.parallel_config.model_parallel as :{uargs.world_size}")

    config.model.model_config.quant = ''

    helper = MFLlama2Helper(config)
    network = helper.create_network()
    config = helper.mf_config
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    logger.info('Quantize-ing network...')
    start = time.time()
    network = quant_network(network, helper, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                            ds_path=uargs.dataset_path, ds_type=uargs.dataset_type)
    logger.info(f'Quant Network cost time is {time.time() - start} s.')
    logger.info('Saving checkpoint...')
    start = time.time()
    save_ckpt_path = os.path.join(config.output_dir, "w8a8_ckpt")
    try:
        rank_id = get_rank()
    except RuntimeError:
        rank_id = 0
    save_path = os.path.join(save_ckpt_path, f"rank_{rank_id}")
    os.makedirs(save_path, exist_ok=True)
    ms.save_checkpoint(network.parameters_dict(), os.path.join(save_path, "w8a8.ckpt"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x)
    logger.info(f'Save checkpoint cost time is {time.time() - start} s.')
    logger.info(f'Checkpoint saved to {save_path}...')
