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
"""Quant network."""
import os
import argparse
import time

import mindspore as ms
from mindspore import dtype as msdtype
from mindspore.communication import get_rank
from mindformers import MindFormerConfig
from mindspore_gs.ptq import PTQMode, PTQConfig, OutliersSuppressionType
from mindspore_gs.common import BackendTarget, logger
from mindspore_gs.ptq import RoundToNearest as RTN
from mindspore_gs.ptq.smooth_quant import SmoothQuant as SQ
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.datasets import get_datasets
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper, MFParallelLlama2Helper


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--approach', '-q', type=str, required=True,
                        help="Available: rtn-a16w8, rtn-c8, smooth_quant, ptq")
    parser.add_argument('--dataset_type', '-t', type=str, required=False)
    parser.add_argument('--dataset_path', '-s', type=str, required=False)

    parser.add_argument('--weight_quant_dtype', '-w', type=str, default='none', help="Available: 'int8', 'none'")
    parser.add_argument('--act_quant_dtype', '-a', type=str, default='none', help="Available: 'int8', 'none'")
    parser.add_argument('--kvcache_quant_dtype', '-k', type=str, default='none', help="Available: 'int8', 'none'")
    parser.add_argument('--outliers_suppression', '-o', type=str, default='none', help="Available: 'smooth', 'none'")
    parser.add_argument('--opname_blacklist', '-b', type=str, nargs='*',
                        help="A list of model layers not to convert, set blacklist when use PTQ algo.")
    parser.add_argument('--world_size', '-ws', type=int, required=True, help="world size, world number")
    parser.add_argument('--load_checkpoint', '-lc', type=str, required=True, help="src checkpoint path")
    parser.add_argument('--output_dir', '-od', type=str, required=True, help="generate quant weight path")


    args = parser.parse_args()

    if args.approach == 'rtn-a16w8':
        logger.info("weight_quant_dtype, act_quant_dtype, kvcache_quant_dtype and outliers_suppression be reset "
                    f"according to approach: {args.approach}.")
        args.weight_quant_dtype = msdtype.int8
        args.act_quant_dtype = None
        args.kvcache_quant_dtype = None
        args.outliers_suppression = OutliersSuppressionType.NONE
        args.opname_blacklist = ['lm_head']
    elif args.approach == 'rtn-c8':
        logger.info("weight_quant_dtype, act_quant_dtype, kvcache_quant_dtype and outliers_suppression be reset "
                    f"according to approach: {args.approach}.")
        args.weight_quant_dtype = None
        args.act_quant_dtype = None
        args.kvcache_quant_dtype = msdtype.int8
        args.outliers_suppression = OutliersSuppressionType.NONE
        args.opname_blacklist = []
    elif args.approach == 'smooth_quant':
        logger.info("weight_quant_dtype, act_quant_dtype, kvcache_quant_dtype and outliers_suppression be reset "
                    f"according to approach: {args.approach}.")
        args.weight_quant_dtype = msdtype.int8
        args.act_quant_dtype = msdtype.int8
        args.kvcache_quant_dtype = None
        args.outliers_suppression = OutliersSuppressionType.SMOOTH
        args.opname_blacklist = ['lm_head', 'w2']
    elif args.approach == 'ptq':
        def dtype_formatter(name: str):
            if name == 'int8':
                return msdtype.int8
            return None

        args.weight_quant_dtype = dtype_formatter(args.weight_quant_dtype)
        args.act_quant_dtype = dtype_formatter(args.act_quant_dtype)
        args.kvcache_quant_dtype = dtype_formatter(args.kvcache_quant_dtype)
        args.outliers_suppression = OutliersSuppressionType.SMOOTH if args.outliers_suppression == 'smooth' \
            else OutliersSuppressionType.NONE
        if args.opname_blacklist:
            args.opname_blacklist = args.opname_blacklist
        else:
            args.opname_blacklist = []
    else:
        raise ValueError(f"Unsupported approach: {args.approach}")

    logger.info(f"quant args: {args}")
    return args


def create_ptq(uargs_, backend=BackendTarget.ASCEND):
    """Create ptq algorithm."""
    start_time = time.time()
    approach = uargs_.approach
    cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=backend, weight_quant_dtype=uargs_.weight_quant_dtype,
                    act_quant_dtype=uargs_.act_quant_dtype, kvcache_quant_dtype=uargs_.kvcache_quant_dtype,
                    outliers_suppression=uargs_.outliers_suppression, opname_blacklist=uargs_.opname_blacklist)
    if approach == 'rtn-c8':
        logger.info("Use RoundToNearest(KVCacheInt8) algo to quant network and weight.")
        ptq = RTN(config=cfg)
    elif approach == 'smooth_quant':
        logger.info("Use SmoothQuant(W8A8) algo to quant network and weight.")
        ptq = SQ(config=cfg)
    elif approach == 'rtn-a16w8':
        logger.info("Use RoundToNearest(W8A16) algo to quant network and weight.")
        ptq = RTN(config=cfg)
    elif approach == 'ptq':
        logger.info("Use ptq algo to quant network and weight.")
        ptq = PTQ(config=cfg)
    else:
        raise ValueError(f"uargs.approach = {uargs_.approach} is unexpected, Available: w8a16, w8a8, c8, ptq.")
    logger.info(f'Create quantizer cost time is {time.time() - start_time} s.')
    return ptq


def create_ds(network_helper, ds_path, ds_type, approach):
    """Create datasets."""
    if approach in ['rtn-c8', 'smooth_quant', 'ptq', 'omni_quant']:
        start_time = time.time()
        if not ds_path:
            raise ValueError(f"Please provide dataset_path when approach is {approach}.")
        if not ds_type:
            raise ValueError(f"Please provide dataset_type when approach is {approach}.")
        bs_ = network_helper.get_spec('batch_size')
        seq_ = network_helper.get_spec('seq_length')
        max_decode_length = network_helper.get_spec('max_decode_length')
        ignore_token_id = network_helper.get_spec('ignore_token_id')
        tokenizer = network_helper.create_tokenizer()
        ds = get_datasets(ds_type, ds_path, "train", bs_, seq_, max_decode_length, tokenizer, ignore_token_id, 1,
                          False, n_samples=200)
        logger.info(f'Create datasets cost time is {time.time() - start_time} s.')
        return ds
    return None


def quant_net(net, network_helper, ptq, ds):
    """Quant network with algorithm."""
    quant_start = time.time()
    logger.info('Quantize-ing network...')
    start_time = time.time()
    ptq.apply(net, network_helper, ds)
    logger.info(f'Apply PTQ cost time is {time.time() - start_time} s.')
    start_time = time.time()
    net.phase = "quant_convert"
    ptq.convert(net)
    logger.info(f'Convert to real quantize cost time is {time.time() - start_time} s.')
    logger.info(f'Quant Network cost total time is {time.time() - quant_start} s.')
    return net


if __name__ == "__main__":
    uargs = get_args()
    algo = create_ptq(uargs)
    msconfig = MindFormerConfig(uargs.config_path)
    if uargs.load_checkpoint:
        msconfig.load_checkpoint = uargs.load_checkpoint
    if uargs.output_dir:
        msconfig.output_dir = uargs.output_dir
    if uargs.world_size:
        if msconfig.parallel_config.model_parallel != uargs.world_size:
            logger.info(
                f"world_size is {uargs.world_size}, not equal to \
                config.parallel_config.model_parallel:{msconfig.parallel_config.model_parallel}")
            msconfig.parallel_config.model_parallel = uargs.world_size
            logger.info(f"reset config.parallel_config.model_parallel as :{uargs.world_size}")

    msconfig.model.model_config.quantization_config = ''
    msconfig.context.mode = 1

    if msconfig.model.arch.type == "LlamaForCausalLM":
        helper = MFLlama2Helper(msconfig)
    elif msconfig.model.arch.type == "ParallelLlamaForCausalLM":
        helper = MFParallelLlama2Helper(msconfig)
    else:
        err_msg = f"Unsupported network arch: {msconfig.model.arch}, please check model.arch in yaml config, " \
                  f"only support LlamaForCausalLM and ParallelLlamaForCausalLM now"
        raise ValueError(err_msg)
    datasets = create_ds(helper, uargs.dataset_path, uargs.dataset_type, approach=uargs.approach)
    start = time.time()
    logger.info('Creating network...')
    network = helper.create_network()
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    network = quant_net(network, helper, algo, datasets)
    logger.info('Saving checkpoint...')
    start = time.time()
    try:
        rank_id = get_rank()
    except RuntimeError:
        rank_id = 0
    save_ckpt_path = os.path.join(helper.mf_config.output_dir)
    save_path = os.path.join(save_ckpt_path, f"rank_{rank_id}")
    os.makedirs(save_path, exist_ok=True)
    ms.save_checkpoint(network.parameters_dict(), os.path.join(save_path, f"{uargs.approach}.ckpt"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x)
    logger.info(f'Save checkpoint cost time is {time.time() - start} s.')
    logger.info(f'Checkpoint saved to {save_path}...')
