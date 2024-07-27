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

import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.nn import DistributedGradReducer

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


class TestData:
    """
    generate a test dataset
    """
    def __init__(self, input_data, label_data, attn_mask=None):
        super().__init__()
        self.input_data = input_data
        self.data_size = self.input_data.shape
        self.label_data = label_data
        self.attn_mask = attn_mask

    def __getitem__(self, index):
        return (Tensor(self.input_data[index]), Tensor(self.label_data[index]), Tensor(self.attn_mask))

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
    if reduce_grad and get_dp_world_size() > 1:
        grad_reducer = DistributedGradReducer(optimizer.parameters, group=get_dp_group())
    all_loss = []
    for epoch in range(epoch_num):
        step = 0
        for data in dataset:
            input_ids, labels, attn_mask = data
            if with_attn_input:
                loss, grads = grad_func(input_ids, attn_mask, labels)
            else:
                loss, grads = grad_func(input_ids, labels)
            if reduce_grad and get_dp_world_size() > 1:
                print("reduce gradients on group {}".format(get_dp_group()))
                grads = grad_reducer(grads)
            loss = ops.depend(loss, optimizer(grads))
            print("Epoch {}, step {}, loss {}".format(epoch, step, loss))
            step += 1
            all_loss.append(loss.asnumpy())

    if save_ckpt_path is not None:
        ms.save_checkpoint(network, save_ckpt_path)
    return all_loss


def generate_ckpt(hidden_size, module_type, num_layers=2, kv_hidden_size=None):
    """ generate graph mode module checkpoints """
    ms.set_seed(1024)
    if not kv_hidden_size:
        kv_hidden_size = hidden_size
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
            Tensor(np.random.random((hidden_size+2*kv_hidden_size, hidden_size)), mstype.float32),
            name=param_name
            )

        # generate attention.w_qkv.bias
        param_name = prefix + '{}attention.w_qkv.bias'.format(str(i) + '.' if has_layer_index else '')
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((hidden_size+2*kv_hidden_size)), mstype.float32),
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


def transform_transformerlayer_params(params, hidden_size, kv_hidden_size=None, prefix=''):
    """
    transform transformerlayer parameters.
    """
    if not kv_hidden_size:
        kv_hidden_size = hidden_size
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
            q = param_t[:, :hidden_size]
            k = param_t[:, hidden_size:hidden_size+kv_hidden_size]
            v = param_t[:, hidden_size+kv_hidden_size:]
            q_start = tp_rank * (q.shape[1] // tp_world_size)
            q_end = (tp_rank+1) * (q.shape[1] // tp_world_size)
            kv_start = tp_rank * (k.shape[1] // tp_world_size)
            kv_end = (tp_rank+1) * (k.shape[1] // tp_world_size)
            new_param = np.concatenate([q[:, q_start:q_end], k[:, kv_start:kv_end], v[:, kv_start:kv_end]], axis=-1)
            new_params[prefix+name.replace("w_qkv.", "qkv_proj.")] = ms.Parameter(ms.Tensor(new_param))
        if 'w_qkv.bias' in name:
            param = param.asnumpy()
            q = param[:hidden_size]
            k = param[hidden_size:hidden_size+kv_hidden_size]
            v = param[hidden_size+kv_hidden_size:]
            q_start = tp_rank * (q.shape[0] // tp_world_size)
            q_end = (tp_rank+1) * (q.shape[0] // tp_world_size)
            kv_start = tp_rank * (k.shape[0] // tp_world_size)
            kv_end = (tp_rank+1) * (k.shape[0] // tp_world_size)
            new_param = np.concatenate([q[q_start:q_end], k[kv_start:kv_end], v[kv_start:kv_end]], axis=0)
            new_params[prefix+name.replace("w_qkv", "qkv_proj")] = ms.Parameter(ms.Tensor(new_param))
        if 'mapping.weight' in name:
            start = tp_rank * (param.shape[1] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[1] // tp_world_size)
            new_param = param[:, start: end]
            new_params[prefix+name] = ms.Parameter(new_param)
        if 'mapping.bias' in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start: end]
            new_params[prefix+name] = ms.Parameter(new_param)
        if 'projection.weight' in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start: end]
            new_params[prefix+name] = ms.Parameter(new_param)
        if 'projection.bias' in name:
            new_param = param
            new_params[prefix+name] = ms.Parameter(new_param)

    return new_params