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
"""some utility functions"""
from typing import List, Dict

import glob
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.nn import DistributedGradReducer

from mindformers import init_context
from mindformers.experimental.distri_cores.create_comm import (
    get_dp_group,
    get_dp_world_size,
    get_tp_rank,
    get_tp_world_size,
)


def get_res(file, pattern):
    """
    find pattern from file
    """
    with open(file, "r") as f:
        lines = f.readlines()
        for l in reversed(lines):
            if pattern in l:
                return l
    return None


def build_context(config):
    """
    build mindspore context using config
    """
    print("..........Init Context..........")
    if config.use_parallel and config.parallel_config.pipeline_stage > 1:
        config.parallel.pipeline_stages = config.parallel_config.pipeline_stage
    local_rank, device_num = init_context(
        use_parallel=config.use_parallel,
        context_config=config.context,
        parallel_config=config.parallel,
    )

    return local_rank, device_num


class TestData:
    """
    generate a test dataset
    """
    def __init__(self, data_size=None, input_data=None, label_data=None, with_attn_mask=False):
        super().__init__()
        self.with_attn_mask = with_attn_mask
        if input_data is not None:
            assert label_data is not None
            self.input_data = input_data
            self.data_size = self.input_data.shape
        else:
            self.input_data = np.random.random(data_size).astype(np.float32)
            self.data_size = self.input_data.shape
        if label_data is not None:
            assert input_data is not None
            self.label_data = label_data
        else:
            self.label_data = np.zeros(self.data_size[:2]).astype(np.float32)
        for i in range(self.data_size[0]):
            self.label_data[i][0] = 1
        seq_length = self.data_size[1]
        if self.with_attn_mask:
            self.attention_mask = np.tril(np.ones(shape=(1, seq_length, seq_length))).astype(np.uint8)

    def __getitem__(self, index):
        if self.with_attn_mask:
            return (Tensor(self.input_data[index]), Tensor(self.label_data[index]), Tensor(self.attention_mask))
        return (Tensor(self.input_data[index]), Tensor(self.label_data[index]))

    def __len__(self):
        return self.input_data.shape[0]


def train(epoch_num, dataset, network, optimizer, save_ckpt_path=None, with_attn_input=False, reduce_grad=True):
    """
    define a train process
    """
    network.set_train()
    grad_func = ops.value_and_grad(
        network, grad_position=None, weights=optimizer.parameters
    )
    if reduce_grad and ms.get_context("mode") == ms.PYNATIVE_MODE and get_dp_world_size() > 1:
        grad_reducer = DistributedGradReducer(optimizer.parameters, group=get_dp_group())
    all_loss = []
    for epoch in range(epoch_num):
        step = 0
        for data in dataset:
            if with_attn_input:
                input_ids, labels, attn_mask = data
                loss, grads = grad_func(input_ids, attn_mask, labels)
            else:
                input_ids, labels = data
                loss, grads = grad_func(input_ids, labels)
            if reduce_grad and ms.get_context("mode") == ms.PYNATIVE_MODE and get_dp_world_size() > 1:
                print("reduce gradients on group {}".format(get_dp_group()))
                grads = grad_reducer(grads)
            loss = ops.depend(loss, optimizer(grads))
            print("Epoch {}, step {}, loss {}".format(epoch, step, loss))
            step += 1
            all_loss.append(loss.asnumpy())

    if save_ckpt_path is not None:
        ms.save_checkpoint(network, save_ckpt_path)
    return all_loss

def transform_mlp_golden_params_to_pynative_params(golden_params: dict):
    """
    transform golden mlp params to pynative params
    """
    tp_rank = get_tp_rank()
    tp_world_size = get_tp_world_size()
    pynative_params = {}
    for name, param in golden_params.items():
        new_param = param
        if 'mapping.weight' in name:
            start = tp_rank * (param.shape[1] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[1] // tp_world_size)
            new_param = param[:, start: end]
        if 'projection.weight' in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start: end]
        if 'mapping.bias' in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start: end]
        pynative_params[name] = ms.Parameter(new_param)

    return pynative_params


def transform_linear_params(params, linear_type):
    """
    transform linear parameters.
    """
    tp_rank = get_tp_rank()
    tp_world_size = get_tp_world_size()
    transformed_params = {}
    for name, param in params.items():
        if linear_type == 'rowparallellinear':
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            if name == "linear.weight":
                new_param = param[start:end, :]
            elif name == "linear.bias":
                new_param = param
        elif linear_type == 'columnparallellinear':
            if name == "linear.weight":
                start = tp_rank * (param.shape[1] // tp_world_size)
                end = (tp_rank + 1) * (param.shape[1] // tp_world_size)
                new_param = param[:, start:end]
            elif name == "linear.bias":
                start = tp_rank * (param.shape[0] // tp_world_size)
                end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
                new_param = param[start:end]
        transformed_params[name] = ms.Parameter(new_param)

    return transformed_params


def generate_ckpt(hidden_size, module_type, num_layers=2):
    """ generate graph mode module checkpoints """
    ms.set_seed(1024)
    prefix = ''
    has_layer_index = False
    if module_type == 'transformer':
        prefix = ''
        has_layer_index = True
    if module_type == 'transformerlayer':
        prefix = 'layer.'
    if module_type in ['attention', 'mlp']:
        prefix = ''

    param_dict = {}
    for i in range(num_layers):
        # generate ffn_norm.weight
        param_name = prefix + '{}ffn_norm.weight'.format(str(i) + '.' if has_layer_index else '')
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((hidden_size)), mstype.float32),
            name=param_name
            )

        # generate attention_norm.weight
        param_name = prefix + '{}attention_norm.weight'.format(str(i) + '.' if has_layer_index else '')
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((hidden_size)), mstype.float32),
            name=param_name
            )

        # generate attention.w_qkv.weight
        param_name = prefix + '{}attention.w_qkv.weight'.format(str(i) + '.' if has_layer_index else '')
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((3*hidden_size, hidden_size)), mstype.float32),
            name=param_name
            )

        # generate attention.w_qkv.bias
        param_name = prefix + '{}attention.w_qkv.bias'.format(str(i) + '.' if has_layer_index else '')
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((3*hidden_size)), mstype.float32),
            name=param_name
            )

        # generate attention.wo.weight
        param_name = prefix + '{}attention.wo.weight'.format(str(i) + '.' if has_layer_index else '')
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((hidden_size, hidden_size)), mstype.float32),
            name=param_name
            )

        # generate mlp.mapping.weight
        param_name = prefix + '{}mlp.mapping.weight'.format(str(i) + '.' if has_layer_index else '')
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((hidden_size, 4*hidden_size)), mstype.float32),
            name=param_name
            )

        # generate mlp.mapping.bias
        param_name = prefix + '{}mlp.mapping.bias'.format(str(i) + '.' if has_layer_index else '')
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((4*hidden_size)), mstype.float32),
            name=param_name
            )

        # generate mlp.projection.weight
        param_name = prefix + '{}mlp.projection.weight'.format(str(i) + '.' if has_layer_index else '')
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((4*hidden_size, hidden_size)), mstype.float32),
            name=param_name
            )

        # generate mlp.projection.bias
        param_name = prefix + '{}mlp.projection.bias'.format(str(i) + '.' if has_layer_index else '')
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((hidden_size)), mstype.float32),
            name=param_name
            )

    return param_dict


def transform_transformerlayer_params(params, prefix=''):
    """
    transform transformerlayer parameters.
    """
    tp_rank = get_tp_rank()
    tp_world_size = get_tp_world_size()
    new_params = {}
    for name, param in params.items():
        if "ffn_norm" in name:
            new_param = param
            new_params[prefix+name.replace("ffn_norm", "post_attention_norm")] = ms.Parameter(new_param)
        if "attention_norm" in name:
            new_param = param
            new_params[prefix+name.replace("attention_norm", "input_norm")] = ms.Parameter(new_param)
        if 'wo.weight' in name:
            param_t = param.asnumpy().transpose()
            start = tp_rank * (param_t.shape[0] // tp_world_size)
            end = (tp_rank+1) * (param_t.shape[0] // tp_world_size)
            new_param = param_t[start:end, :]
            new_params[prefix+name.replace("wo", "out_proj")] = ms.Parameter(new_param)
        if 'w_qkv.weight' in name:
            param_t = param.asnumpy().transpose()
            q = param_t[:, :param_t.shape[1]//3]
            k = param_t[:, param_t.shape[1]//3:2*param_t.shape[1]//3]
            v = param_t[:, 2*param_t.shape[1]//3:]
            start = tp_rank * (q.shape[1] // tp_world_size)
            end = (tp_rank+1) * (q.shape[1] // tp_world_size)
            new_param = np.concatenate([q[:, start:end], k[:, start:end], v[:, start:end]], axis=-1)
            new_params[prefix+name.replace("w_qkv.", "qkv_proj.")] = ms.Parameter(ms.Tensor(new_param))
        if 'w_qkv.bias' in name:
            param = param.asnumpy()
            q = param[:param.shape[0]//3]
            k = param[param.shape[0]//3:2*param.shape[0]//3]
            v = param[2*param.shape[0]//3:]
            start = tp_rank * (q.shape[0] // tp_world_size)
            end = (tp_rank+1) * (q.shape[0] // tp_world_size)
            new_param = np.concatenate([q[start:end], k[start:end], v[start:end]], axis=0)
            new_params[prefix+name.replace("w_qkv", "qkv_proj")] = ms.Parameter(ms.Tensor(new_param))
        # w3 -> mapping
        if 'w3.weight' in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start: end].T
            new_params[prefix+name.replace("w3", "mapping")] = ms.Parameter(new_param)
        if 'w3.bias' in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start: end]
            new_params[prefix+name.replace("w3", "mapping")] = ms.Parameter(new_param)
        # w2 -> projection
        if 'w2.weight' in name:
            start = tp_rank * (param.shape[1] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[1] // tp_world_size)
            new_param = param[:, start: end].T
            new_params[prefix+name.replace("w2", "projection")] = ms.Parameter(new_param)
        if 'w2.bias' in name:
            new_param = param
            new_params[prefix+name.replace("w2", "projection")] = ms.Parameter(new_param)
        # w1 -> gating
        if 'w1.weight' in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start: end].T
            new_params[prefix+name.replace("w1", "gating")] = ms.Parameter(new_param)
        if 'w1.bias' in name:
            new_param = param
            new_params[prefix+name.replace("w1", "gating")] = ms.Parameter(new_param)


    return new_params


def transform_llama_params(params):
    """
    transform transformerlayer parameters.
    """
    tp_rank = get_tp_rank()
    tp_world_size = get_tp_world_size()
    new_params = {}
    for name, param in params.items():
        if "ffn_norm" in name:
            new_param = param
            new_params[prefix+name.replace("ffn_norm", "post_attention_norm")] = ms.Parameter(new_param)
        if "attention_norm" in name:
            new_param = param
            new_params[prefix+name.replace("attention_norm", "input_norm")] = ms.Parameter(new_param)
        if 'wo.weight' in name:
            param_t = param.asnumpy().transpose()
            start = tp_rank * (param_t.shape[0] // tp_world_size)
            end = (tp_rank+1) * (param_t.shape[0] // tp_world_size)
            new_param = param_t[start:end, :]
            new_params[prefix+name.replace("wo", "out_proj")] = ms.Parameter(new_param)
        if 'w_qkv.weight' in name:
            param_t = param.asnumpy().transpose()
            q = param_t[:, :param_t.shape[1]//3]
            k = param_t[:, param_t.shape[1]//3:2*param_t.shape[1]//3]
            v = param_t[:, 2*param_t.shape[1]//3:]
            start = tp_rank * (q.shape[1] // tp_world_size)
            end = (tp_rank+1) * (q.shape[1] // tp_world_size)
            new_param = np.concatenate([q[:, start:end], k[:, start:end], v[:, start:end]], axis=-1)
            new_params[prefix+name.replace("w_qkv.", "qkv_proj.")] = ms.Parameter(ms.Tensor(new_param))
        # w3 -> mapping
        if 'w3.weight' in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start: end].T
            new_params[prefix+name.replace("w3", "mapping")] = ms.Parameter(new_param)
        # w2 -> projection
        if 'w2.weight' in name:
            start = tp_rank * (param.shape[1] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[1] // tp_world_size)
            new_param = param[:, start: end].T
            new_params[prefix+name.replace("w2", "projection")] = ms.Parameter(new_param)
        # w1 -> gating
        if 'w1.weight' in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start: end].T
            new_params[prefix+name.replace("w1", "gating")] = ms.Parameter(new_param)

    return new_params


def transform_mlp_has_gate_golden_params_to_pynative_params(golden_params: dict):
    """
    transform golden mlp params to pynative params
    """
    tp_rank = get_tp_rank()
    tp_world_size = get_tp_world_size()

    print(f"golden to pynative ckpt map:")
    pynative_params = {}
    for name, param in golden_params.items():
        if "w1" in name: # w1 -> gating
            pynative_name = 'mlp.gating.weight'
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = ops.t(param[start: end])
        if "w2" in name: # w2 -> projection
            pynative_name = 'mlp.projection.weight'
            start = tp_rank * (param.shape[1] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[1] // tp_world_size)
            new_param = ops.t(param[:, start: end])
        if "w3" in name: # w3 -> mapping
            pynative_name = 'mlp.mapping.weight'
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = ops.t(param[start: end])
        pynative_params[pynative_name] = ms.Parameter(new_param)
        print(f"{name} {param.shape} -> {pynative_name} {pynative_params[pynative_name].shape}", flush=True)

    return pynative_params


def transform_vocab_embedding_golden_params_to_pynative_params(golden_params: dict):
    """
    transform golden vocab embedding params to pynative params
    """
    tp_rank = get_tp_rank()
    tp_world_size = get_tp_world_size()
    pynative_params = {}
    param = golden_params["embedding.embedding_table"]
    pynative_name = "embedding.weight"
    start = tp_rank * (param.shape[0] // tp_world_size)
    end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
    new_param = param[start:end, :]
    pynative_params[pynative_name] = ms.Parameter(new_param, name=pynative_name)

    return pynative_params


def rearrange_the_ckpt_files(parent_dir, world_size):
    """rearrage the ckpt files into its parent directory"""
    import os
    import shutil
    if not os.path.exists(parent_dir):
        print(f"The directory {parent_dir} does not exist.")
    else:
        for rank in range(world_size):
            rank_dir = os.path.join(parent_dir, f'rank_{rank}')
            if os.path.exists(rank_dir):
                src_strategy_files = glob.glob(os.path.join(rank_dir, "*.ckpt"))
                for src_strategy_file in src_strategy_files:
                    new_file_name = f'strategy_{rank}.ckpt'
                    new_file_path = os.path.join(parent_dir, new_file_name)
                    shutil.move(src_strategy_file, new_file_path)
                    shutil.rmtree(rank_dir)
                    print(f"Moved and renamed {src_strategy_file} to {new_file_path}")
            else:
                print(f"Rank directory {rank_dir} does not exist.")


def transform_moe_golden_params_to_pynative_params(golden_params: Dict[str, Tensor],
                                                   local_expert_indices: List[int]):
    """
    transform golden moe params to pynative params
    map_dict = {"w1": "gating",
                "w2": "projection",
                "w3": "mapping"}
    """

    print(f"golden to pynative ckpt map:")
    pynative_params = {}
    for name, param in golden_params.items():
        if "feed_forward.ffn" in name:
            for i, local_expert_id in enumerate(local_expert_indices):
                if "w1" in name: # w1 -> gating
                    pynative_name = f'moe.experts.local_experts.{i}.gating.weight'
                if "w2" in name: # w2 -> projection
                    pynative_name = f'moe.experts.local_experts.{i}.projection.weight'
                if "w3" in name: # w3 -> mapping
                    pynative_name = f'moe.experts.local_experts.{i}.mapping.weight'
                if len(param.shape) == 3:
                    pynative_params[pynative_name] = ms.Parameter(param[local_expert_id].T)
                    print(f"{name} {param[local_expert_id].shape} -> "+\
                          f"{pynative_name} {pynative_params[pynative_name].shape}", flush=True)
                elif len(param.shape) == 2:
                    pynative_params[pynative_name] = ms.Parameter(param.t)
                    print(f"{name} {param.shape} -> {pynative_name} {pynative_params[pynative_name].shape}", flush=True)
        if "feed_forward.router.dense.weight" in name:
            pynative_name = 'moe.router.gating.weight'
            pynative_params[pynative_name] = ms.Parameter(param)
            print(f"{name} {param.shape} -> {pynative_name} {pynative_params[pynative_name].shape}", flush=True)

    return pynative_params


def transform_sequential_mlp_golden_params_to_pynative_params(golden_params: Dict[str, Tensor],
                                                              local_expert_indices: List[int]):
    """
    transform golden moe params to pynative params
    map_dict = {"w1": "gating",
                "w2": "projection",
                "w3": "mapping"}
    """

    print(f"golden to pynative ckpt map:")
    pynative_params = {}
    for name, param in golden_params.items():
        for i, local_expert_id in enumerate(local_expert_indices):
            if f"{local_expert_id}.w1" in name: # w1 -> gating
                pynative_name = f'mlp.local_experts.{i}.gating.weight'
                pynative_params[pynative_name] = ms.Parameter(param.T)
                print(f"{name} {param.shape} -> {pynative_name} {pynative_params[pynative_name].shape}", flush=True)
            if f"{local_expert_id}.w2" in name: # w2 -> projection
                pynative_name = f'mlp.local_experts.{i}.projection.weight'
                pynative_params[pynative_name] = ms.Parameter(param.T)
                print(f"{name} {param.shape} -> {pynative_name} {pynative_params[pynative_name].shape}", flush=True)
            if f"{local_expert_id}.w3" in name: # w3 -> mapping
                pynative_name = f'mlp.local_experts.{i}.mapping.weight'
                pynative_params[pynative_name] = ms.Parameter(param.T)
                print(f"{name} {param.shape} -> {pynative_name} {pynative_params[pynative_name].shape}", flush=True)

    return pynative_params
