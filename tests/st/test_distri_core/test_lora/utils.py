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
"""Test LoRA utils"""
import numpy as np

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.nn import DistributedGradReducer

from mindformers.experimental.distri_cores.create_comm import (
    get_dp_group,
    get_dp_world_size,
    get_tp_rank,
    get_tp_world_size,
    get_tp_group
)


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


def train(epoch_num, dataset, network, optimizer, save_ckpt_path=None, with_attn_input=False, reduce_grad=True,
          use_sequence_parallel=False):
    """
    define a train process
    """
    network.set_train()
    grad_func = ops.value_and_grad(
        network, grad_position=None, weights=optimizer.parameters
    )
    if reduce_grad and ms.get_context("mode") == ms.PYNATIVE_MODE and get_dp_world_size() > 1:
        grad_reducer = DistributedGradReducer(optimizer.parameters, group=get_dp_group())
    if use_sequence_parallel:
        print('grad reducer: use_sequence_parallel')
        sp_reduce_lora_params = ['qkv_proj.lora_a', 'out_proj.lora_b', 'mapping.lora_a', 'projection.lora_b']
        seq_parameters = ()
        for i, param in enumerate(optimizer.parameters):
            if any(p in param.name for p in sp_reduce_lora_params):
                seq_parameters = seq_parameters + (param,)
        grad_reducers_sp = DistributedGradReducer(seq_parameters, group=get_tp_group())

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
            if use_sequence_parallel:
                sp_grads = ()
                for i, param in enumerate(optimizer.parameters):
                    if any(p in param.name for p in sp_reduce_lora_params):
                        sp_grads = sp_grads + (grads[i],)
                sp_grads = grad_reducers_sp(sp_grads)
                print('reduce sp grads')

                grads_after_reduce = ()
                c = 0
                for i, param in enumerate(optimizer.parameters):
                    if any(p in param.name for p in sp_reduce_lora_params):
                        grads_after_reduce += (sp_grads[c],)
                        c += 1
                    else:
                        grads_after_reduce += (grads[i],)
                grads = grads_after_reduce
            loss = ops.depend(loss, optimizer(grads))
            print("Epoch {}, step {}, loss {}".format(epoch, step, loss))
            step += 1
            all_loss.append(loss.asnumpy())

    if save_ckpt_path is not None:
        ms.save_checkpoint(network, save_ckpt_path)
    return all_loss


def transform_transformerlayer_params(params, hidden_size, kv_hidden_size=None):
    """
    transform transformerlayer parameters.
    """
    if not kv_hidden_size:
        kv_hidden_size = hidden_size
    tp_rank = get_tp_rank()
    tp_world_size = get_tp_world_size()
    new_params = {}
    for name, param in params.items():
        if "input_norm" in name or "post_attention_norm" in name:
            new_param = param
            new_params[name] = ms.Parameter(new_param)
        if "qkv_proj.weight" in name or 'qkv_proj.lora_b' in name:
            param = param.asnumpy()
            q = param[:hidden_size, :]
            k = param[hidden_size:hidden_size + kv_hidden_size, :]
            v = param[hidden_size + kv_hidden_size:, :]
            q_start = tp_rank * (q.shape[0] // tp_world_size)
            q_end = (tp_rank + 1) * (q.shape[0] // tp_world_size)
            kv_start = tp_rank * (k.shape[0] // tp_world_size)
            kv_end = (tp_rank + 1) * (k.shape[0] // tp_world_size)
            new_param = np.concatenate([q[q_start:q_end, :], k[kv_start:kv_end, :], v[kv_start:kv_end, :]], axis=0)
            new_params[name] = ms.Parameter(ms.Tensor(new_param))
        if 'qkv_proj.bias' in name:
            param = param.asnumpy()
            q = param[:hidden_size]
            k = param[hidden_size:hidden_size + kv_hidden_size]
            v = param[hidden_size + kv_hidden_size:]
            q_start = tp_rank * (q.shape[0] // tp_world_size)
            q_end = (tp_rank + 1) * (q.shape[0] // tp_world_size)
            kv_start = tp_rank * (k.shape[0] // tp_world_size)
            kv_end = (tp_rank + 1) * (k.shape[0] // tp_world_size)
            new_param = np.concatenate([q[q_start:q_end], k[kv_start:kv_end], v[kv_start:kv_end]], axis=0)
            new_params[name] = ms.Parameter(ms.Tensor(new_param))
        if 'out_proj.weight' in name or 'out_proj.lora_a' in name:
            param = param.asnumpy()
            start = tp_rank * (param.shape[1] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[1] // tp_world_size)
            new_param = param[:, start:end]
            new_params[name] = ms.Parameter(new_param)
        if 'mapping.weight' in name or 'mapping.lora_b' in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start: end]
            new_params[name] = ms.Parameter(new_param)
        if 'mapping.bias' in name:
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start: end]
            new_params[name] = ms.Parameter(new_param)
        if 'projection.weight' in name or 'projection.lora_a' in name:
            start = tp_rank * (param.shape[1] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[1] // tp_world_size)
            new_param = param[:, start: end]
            new_params[name] = ms.Parameter(new_param)
        if 'projection.bias' in name:
            new_param = param
            new_params[name] = ms.Parameter(new_param)
        if 'mapping.lora_a' in name or 'qkv_proj.lora_a' in name:
            new_param = param
            new_params[name] = ms.Parameter(new_param)
        if 'projection.lora_b' in name or 'out_proj.lora_b' in name:
            new_param = param
            new_params[name] = ms.Parameter(new_param)

    return new_params
