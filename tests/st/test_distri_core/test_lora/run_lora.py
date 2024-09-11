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
""" Test ParallelTransformer. """

import argparse
import copy
import numpy as np

import mindspore as ms
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.nn import AdamWeightDecay
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init
from mindspore.communication import get_rank
from mindspore.dataset import DistributedSampler

from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel, get_dp_rank
from mindformers.experimental.parallel_core.pynative.transformer import ParallelTransformer
from mindformers.experimental.parallel_core.pynative.transformer.rotary_pos_embedding import RotaryEmbedding
from mindformers.experimental.parallel_core.pynative.config import LoraConfig, TransformerConfig, ModelParallelConfig
from mindformers.experimental.parallel_core.pynative.utils import valid_lora_config
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.tensor_parallel import (
    GatherFromSequenceParallelRegion,
    ScatterToSequenceParallelRegion
)

from utils import transform_transformerlayer_params, train, TestData


def mark_only_lora_as_trainable(network):
    """mark only lora parameters as trainable"""
    for param in network.get_parameters():
        if 'lora' in param.name:
            param.requires_grad = True
        else:
            param.requires_grad = False


class ParallelTransformerNet(Module):
    """ ParallelTransformerNet. """

    def __init__(self, config, with_rope=False, use_sequence_parallel=False):
        super(ParallelTransformerNet, self).__init__()
        self.with_rope = with_rope
        if with_rope:
            self.rope = RotaryEmbedding(config.hidden_size // config.num_heads,
                                        rotary_percent=1.0)
        use_lora = config.lora_config.use_lora
        transformer_config = copy.deepcopy(config)
        if use_lora:
            transformer_config.update_lora_config('transformer')
        self.transformer = ParallelTransformer(config=transformer_config, post_norm=False)
        self.loss = SoftmaxCrossEntropyWithLogits()
        self.use_sequence_parallel = use_sequence_parallel
        self.scatter_to_sp_region = ScatterToSequenceParallelRegion(need_to_swapaxes=False)
        self.gather_from_sp_region = GatherFromSequenceParallelRegion(
            need_to_swapaxes=False, tensor_parallel_output_grad=False
        )

    def construct(self, x, attention_mask, labels):
        """ construct. """
        if self.use_sequence_parallel:
            x = x.swapaxes(0, 1).contiguous()
            x = self.scatter_to_sp_region(x)
            x = x.swapaxes(0, 1).contiguous()
        if self.with_rope:
            emb = self.rope(max_seq_len=x.shape[1])
            output = self.transformer(x, attention_mask, rotary_pos_emb=emb)
        else:
            output = self.transformer(x, attention_mask)
        if self.use_sequence_parallel:
            output = output.swapaxes(0, 1).contiguous()
            output = self.gather_from_sp_region(output)
            output = output.swapaxes(0, 1).contiguous()
        output = ops.sum(output, dim=-1, keepdim=False)
        loss = self.loss(output, labels)
        return loss


def run_parallel_transformer_pretrain():
    """ Test ParallelTransformer pretrain. """
    batch_size = 1
    dataset_size = 10
    num_layers = 2
    seq_length = 16
    num_heads = 4
    hidden_size = 32
    tensor_parallel = 1

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data, with_attn_mask=True)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"])
    dataset = dataset.batch(batch_size)

    lora_config = LoraConfig(use_lora=False)
    parallel_config = ModelParallelConfig(expert_parallel=1, use_sequence_parallel=False)
    config = TransformerConfig(vocab_size=50304,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               hidden_size=hidden_size,
                               ffn_hidden_size=4 * hidden_size,
                               seq_length=seq_length,
                               attention_type='self_attn',
                               use_gqa=False,
                               qkv_has_bias=True,
                               out_proj_has_bias=False,
                               apply_query_key_layer_scaling=True,
                               hidden_dropout_rate=0.0,
                               attention_dropout_rate=0.0,
                               parallel_config=parallel_config,
                               lora_config=lora_config,
                               mask_func_type="attn_mask_add",
                               mlp_has_bias=True,
                               hidden_act='gelu',
                               apply_residual_connection_post_norm=False,
                               normalization='FusedRMSNorm',
                               layernorm_epsilon=1.e-5)
    network = ParallelTransformerNet(config=config, with_rope=False)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None, with_attn_input=True)
    ms.save_checkpoint(network, f'pretrain.ckpt')


def run_parallel_transformer_lora_standalone(target):
    """ Test ParallelTransformer lora. """
    batch_size = 1
    dataset_size = 10
    num_layers = 2
    seq_length = 16
    num_heads = 4
    hidden_size = 32
    tensor_parallel = 1

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data, with_attn_mask=True)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"])
    dataset = dataset.batch(batch_size)

    lora_config = LoraConfig(use_lora=True, target_cells=target)
    parallel_config = ModelParallelConfig(expert_parallel=1)
    config = TransformerConfig(vocab_size=50304,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               hidden_size=hidden_size,
                               ffn_hidden_size=4 * hidden_size,
                               seq_length=seq_length,
                               attention_type='self_attn',
                               use_gqa=False,
                               qkv_has_bias=True,
                               out_proj_has_bias=False,
                               apply_query_key_layer_scaling=True,
                               hidden_dropout_rate=0.0,
                               attention_dropout_rate=0.0,
                               parallel_config=parallel_config,
                               lora_config=lora_config,
                               mask_func_type="attn_mask_add",
                               mlp_has_bias=True,
                               hidden_act='gelu',
                               apply_residual_connection_post_norm=False,
                               normalization='FusedRMSNorm',
                               layernorm_epsilon=1.e-5,
                               compute_dtype='float32')
    rank_id = get_rank()

    pretrain_params = ms.load_checkpoint(f'pretrain.ckpt')
    config = valid_lora_config(config, pretrain_params)

    network = ParallelTransformerNet(config=config, with_rope=False)

    pynative_params = transform_transformerlayer_params(pretrain_params, hidden_size=hidden_size)
    ms.load_param_into_net(network, pynative_params)

    mark_only_lora_as_trainable(network)
    ms.save_checkpoint(network, f'msrun_log_lora_col_row/lora_rank{rank_id}_init.ckpt')

    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None, with_attn_input=True)
    rank_id = get_rank()
    ms.save_checkpoint(network, f'msrun_log_lora_col_row/lora_rank{rank_id}.ckpt')


def run_parallel_transformer_lora_tp2(target, use_sequence_parallel=False):
    """ Test ParallelTransformer lora. """
    batch_size = 1
    dataset_size = 10
    num_layers = 2
    seq_length = 16
    num_heads = 4
    hidden_size = 32
    tensor_parallel = 2

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data, with_attn_mask=True)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"])
    dataset = dataset.batch(batch_size)

    lora_config = LoraConfig(use_lora=True, target_cells=target)
    parallel_config = ModelParallelConfig(expert_parallel=1, use_sequence_parallel=use_sequence_parallel)
    config = TransformerConfig(vocab_size=50304,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               hidden_size=hidden_size,
                               ffn_hidden_size=4 * hidden_size,
                               seq_length=seq_length,
                               attention_type='self_attn',
                               use_gqa=False,
                               qkv_has_bias=True,
                               out_proj_has_bias=False,
                               apply_query_key_layer_scaling=True,
                               hidden_dropout_rate=0.0,
                               attention_dropout_rate=0.0,
                               parallel_config=parallel_config,
                               lora_config=lora_config,
                               mask_func_type="attn_mask_add",
                               mlp_has_bias=True,
                               hidden_act='gelu',
                               apply_residual_connection_post_norm=False,
                               normalization='FusedRMSNorm',
                               layernorm_epsilon=1.e-5,
                               compute_dtype='float32')
    rank_id = get_rank()

    pretrain_params = ms.load_checkpoint(f'msrun_log_lora_col_row/lora_rank0_init.ckpt')
    config = valid_lora_config(config, pretrain_params)

    network = ParallelTransformerNet(config=config, with_rope=False, use_sequence_parallel=use_sequence_parallel)

    pynative_params = transform_transformerlayer_params(pretrain_params, hidden_size=hidden_size)
    ms.load_param_into_net(network, pynative_params)

    mark_only_lora_as_trainable(network)
    if use_sequence_parallel:
        ms.save_checkpoint(network, f'msrun_log_lora_tp_sp_col_row/lora_rank{rank_id}_init.ckpt')
    else:
        ms.save_checkpoint(network, f'msrun_log_lora_tp_col_row/lora_rank{rank_id}_init.ckpt')

    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None, with_attn_input=True, use_sequence_parallel=use_sequence_parallel)
    rank_id = get_rank()
    if use_sequence_parallel:
        ms.save_checkpoint(network, f'msrun_log_lora_tp_sp_col_row/lora_rank{rank_id}.ckpt')
    else:
        ms.save_checkpoint(network, f'msrun_log_lora_tp_col_row/lora_rank{rank_id}.ckpt')


def run_parallel_transformer_lora_dp2(target):
    """ Test ParallelTransformer lora. """
    batch_size = 1
    dataset_size = 10
    num_layers = 2
    seq_length = 16
    num_heads = 4
    hidden_size = 32
    tensor_parallel = 1

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data, with_attn_mask=True)

    sampler = DistributedSampler(num_shards=2, shard_id=get_dp_rank(), shuffle=False)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"], sampler=sampler)
    dataset = dataset.batch(batch_size)

    lora_config = LoraConfig(use_lora=True, target_cells=target)
    parallel_config = ModelParallelConfig(expert_parallel=1, use_sequence_parallel=False)
    config = TransformerConfig(vocab_size=50304,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               hidden_size=hidden_size,
                               ffn_hidden_size=4 * hidden_size,
                               seq_length=seq_length,
                               attention_type='self_attn',
                               use_gqa=False,
                               qkv_has_bias=True,
                               out_proj_has_bias=False,
                               apply_query_key_layer_scaling=True,
                               hidden_dropout_rate=0.0,
                               attention_dropout_rate=0.0,
                               parallel_config=parallel_config,
                               lora_config=lora_config,
                               mask_func_type="attn_mask_add",
                               mlp_has_bias=True,
                               hidden_act='gelu',
                               apply_residual_connection_post_norm=False,
                               normalization='FusedRMSNorm',
                               layernorm_epsilon=1.e-5,
                               compute_dtype='float32'
                               )
    rank_id = get_rank()
    pretrain_params = ms.load_checkpoint(f'msrun_log_lora_col_row/lora_rank0_init.ckpt')
    config = valid_lora_config(config, pretrain_params)

    network = ParallelTransformerNet(config=config, with_rope=False)

    pynative_params = transform_transformerlayer_params(pretrain_params, hidden_size=hidden_size)
    ms.load_param_into_net(network, pynative_params)

    mark_only_lora_as_trainable(network)
    ms.save_checkpoint(network, f'msrun_log_lora_dp_col_row/lora_rank{rank_id}_init.ckpt')

    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None, with_attn_input=True)
    rank_id = get_rank()
    ms.save_checkpoint(network, f'msrun_log_lora_dp_col_row/lora_rank{rank_id}.ckpt')


def run_parallel_transformer_lora_tp2_dp2(target, use_sequence_parallel=False):
    """ Test ParallelTransformer lora. """
    batch_size = 1
    dataset_size = 10
    num_layers = 2
    seq_length = 16
    num_heads = 4
    hidden_size = 32
    tensor_parallel = 2

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data, with_attn_mask=True)

    sampler = DistributedSampler(num_shards=2, shard_id=get_dp_rank(), shuffle=False)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"], sampler=sampler)
    dataset = dataset.batch(batch_size)

    lora_config = LoraConfig(use_lora=True, target_cells=target, col_row_type='col_row')
    parallel_config = ModelParallelConfig(expert_parallel=1, use_sequence_parallel=use_sequence_parallel)
    config = TransformerConfig(vocab_size=50304,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               hidden_size=hidden_size,
                               ffn_hidden_size=4 * hidden_size,
                               seq_length=seq_length,
                               attention_type='self_attn',
                               use_gqa=False,
                               qkv_has_bias=True,
                               out_proj_has_bias=False,
                               apply_query_key_layer_scaling=True,
                               hidden_dropout_rate=0.0,
                               attention_dropout_rate=0.0,
                               parallel_config=parallel_config,
                               lora_config=lora_config,
                               mask_func_type="attn_mask_add",
                               mlp_has_bias=True,
                               hidden_act='gelu',
                               apply_residual_connection_post_norm=False,
                               normalization='FusedRMSNorm',
                               layernorm_epsilon=1.e-5,
                               compute_dtype='float32'
                               )
    rank_id = get_rank()
    pretrain_params = ms.load_checkpoint(f'msrun_log_lora_col_row/lora_rank0_init.ckpt')
    config = valid_lora_config(config, pretrain_params)

    network = ParallelTransformerNet(config=config, with_rope=False, use_sequence_parallel=use_sequence_parallel)

    pynative_params = transform_transformerlayer_params(pretrain_params, hidden_size=hidden_size)
    ms.load_param_into_net(network, pynative_params)

    if use_sequence_parallel:
        log_path = f'msrun_log_lora_dp_tp_sp_{col_row_type}/lora_rank{rank_id}_init.ckpt'
    else:
        log_path = f'msrun_log_lora_dp_tp_{col_row_type}/lora_rank{rank_id}_init.ckpt'
    mark_only_lora_as_trainable(network)
    ms.save_checkpoint(network, log_path)

    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None, with_attn_input=True, use_sequence_parallel=use_sequence_parallel)

    if use_sequence_parallel:
        log_path = f'msrun_log_lora_dp_tp_sp_{col_row_type}/lora_rank{rank_id}.ckpt'
    else:
        log_path = f'msrun_log_lora_dp_tp_{col_row_type}/lora_rank{rank_id}.ckpt'
    ms.save_checkpoint(network, log_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pretrain', action='store_true', help="model pretrain."
    )
    parser.add_argument(
        '--standalone', action='store_true', help="model pretrain."
    )
    parser.add_argument(
        '--col_row_type', default='col_row', help="col row type."
    )
    parser.add_argument(
        '--use_sequence_parallel', action='store_true', help="col row type."
    )
    parser.add_argument(
        '--parallel_strategy', default='dp_tp', help="col row type."
    )

    args, rest_args = parser.parse_known_args()
    col_row_type = args.col_row_type
    if col_row_type == 'col_row':
        target_cells = [
            {'target_cells': [
                '.*.mapping',
                '.*.projection',
                '.*.qkv_proj',
                '.*.out_proj',
            ]
            },
            {'cell': 'transformer.layers.0.attention.qkv_proj',
             'rank': 4,
             'alpha': 16
             },
        ]
    elif col_row_type == 'col':
        target_cells = [
            {'target_cells': [
                '.*.mapping',
                '.*.qkv_proj'
            ]
            },
            {'cell': 'transformer.layers.0.attention.qkv_proj',
             'rank': 4,
             'alpha': 16
             },
        ]
    elif col_row_type == 'row':
        target_cells = [
            {'target_cells': [
                '.*.projection',
                '.*.out_proj'
            ]
            },
            {'cell': 'transformer.layers.0.attention.out_proj',
             'rank': 4,
             'alpha': 16
             },
        ]
    else:
        print('wrong col_row_type!')

    if args.pretrain:
        run_parallel_transformer_pretrain()
    else:
        if args.standalone:
            run_parallel_transformer_lora_standalone(target_cells)
        else:
            if args.parallel_strategy == 'dp_tp':
                run_parallel_transformer_lora_tp2_dp2(target_cells, args.use_sequence_parallel)
            elif args.parallel_strategy == 'dp':
                run_parallel_transformer_lora_dp2(target_cells)
            elif args.parallel_strategy == 'tp':
                run_parallel_transformer_lora_tp2(target_cells, args.use_sequence_parallel)
            else:
                print('wrong parallel_strategy!')
