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
# This file was refer to project:
# https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/PanGu-%CE%B1/utils.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
# ============================================================================
"""Optimizer util."""
import numpy as np
import mindspore
from mindspore import nn, Tensor
from mindspore.ops import operations as P

from mindformers.core.optim import build_optim
from mindformers.core.optim.muon import Muon

np.random.seed(1024)

fc1_weight = np.array([[0.72346634, 0.95608497, 0.4084163, 0.18627149,
                        0.6942514, 0.39767185, 0.24918061, 0.4548748],
                       [0.7203382, 0.19086994, 0.76286614, 0.87920564,
                        0.3169892, 0.9462494, 0.62827677, 0.27504718],
                       [0.3544535, 0.2524781, 0.5370583, 0.8313121,
                        0.6670143, 0.0488653, 0.62225235, 0.7546456],
                       [0.17985944, 0.05106374, 0.31064633, 0.4863033,
                        0.848814, 0.5523157, 0.20295663, 0.7213356]]).astype("float32")

fc1_bias = np.array([0.79708564, 0.13728078, 0.66322654, 0.88128525]).astype("float32")

fc2_weight = np.array([[0.8473515, 0.50923985, 0.42287776, 0.29769543]]).astype("float32")

fc2_bias = np.array([0.09996348]).astype("float32")


def make_fake_data():
    """
    make fake data
    """
    data, label = [], []
    for i in range(20):
        data.append(mindspore.Tensor(np.array(np.ones((2, 8)) * i, dtype=np.float32)))
        label.append(mindspore.Tensor(np.array(np.ones((2, 1)) * (i + 1), dtype=np.float32)))
    return data, label


class NetWithLoss(nn.Cell):
    """
    build net with loss
    """

    def __init__(self, network, loss_fn):
        super().__init__()
        self.network = network
        self.loss = loss_fn

    def construct(self, x, label):
        out = self.network(x)
        loss = self.loss(out, label)
        return loss


class FakeNet(nn.Cell):
    """
    build fake net
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(in_channels=8, out_channels=4, weight_init=Tensor(fc1_weight), bias_init=Tensor(fc1_bias))
        self.fc2 = nn.Dense(in_channels=4, out_channels=1, weight_init=Tensor(fc2_weight), bias_init=Tensor(fc2_bias))
        self.relu = nn.ReLU()
        self.reducemean = P.ReduceMean()

    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        """
        parameter initialization
        """
        self.init_parameters_data()
        for name, m in self.cells_and_names():
            if name == 'fc1':
                m.weight.set_data(Tensor(fc1_weight))
                m.bias.set_data(Tensor(fc1_bias))
            elif name == 'fc2':
                m.weight.set_data(Tensor(fc2_weight))
                m.bias.set_data(Tensor(fc2_bias))


def build_network(opt_config, net, is_group=None, loss_fn=None):
    """
    Construct training
    """
    if is_group is None:
        is_group = False
    if loss_fn is None:
        loss_fn = nn.L1Loss(reduction='mean')
    losses = []
    networkwithloss = NetWithLoss(net, loss_fn)
    networkwithloss.set_train()

    if is_group:
        fc1_params = list(filter(lambda x: 'fc1' in x.name, networkwithloss.trainable_params()))
        fc2_params = list(filter(lambda x: 'fc1' not in x.name, networkwithloss.trainable_params()))
        if opt_config['type'] == 'AdamW':
            params = [{'params': fc1_params, 'weight_decay': 0.01, 'lr': 0.01}, {'params': fc2_params, 'lr': 0.1}]
        else:
            params = [{'params': fc1_params, 'lr': 0.01}, {'params': fc2_params, 'lr': 0.01}]
    else:
        params = networkwithloss.trainable_params()

    opt_config['params'] = params

    net_opt = build_optim(opt_config)
    trainonestepcell = mindspore.nn.TrainOneStepCell(networkwithloss, net_opt)
    data, label = make_fake_data()
    for i in range(20):
        loss = trainonestepcell(data[i], label[i])
        losses.append(loss.asnumpy())
    return np.array(losses), net_opt


default_fc1_weight_adamw_m = (
    np.array([[0.75276935, 0.75276935, 0.75276935, 0.75276935, 0.75276935, 0.75276935, 0.75276935, 0.75276935],
              [0.28740492, 0.28740492, 0.28740492, 0.28740492, 0.28740492, 0.28740492, 0.28740492, 0.28740492],
              [0.12561864, 0.12561864, 0.12561864, 0.12561864, 0.12561864, 0.12561864, 0.12561864, 0.12561864],
              [-0.06905057, -0.06905057, -0.06905057, -0.06905057, -0.06905057, -0.06905057, -0.06905057,
               -0.06905057]], dtype=np.float32)
)

default_fc2_weight_adamw_m = (
    np.array([[6.978479, 7.470356, 5.508465, 5.176325]], dtype=np.float32)
)

default_fc1_weight_adamw_v = (
    np.array([[0.28913346, 0.28913346, 0.28913346, 0.28913346, 0.28913346, 0.28913346, 0.28913346, 0.28913346],
              [0.01420226, 0.01420226, 0.01420226, 0.01420226, 0.01420226, 0.01420226, 0.01420226, 0.01420226],
              [0.00199351, 0.00199351, 0.00199351, 0.00199351, 0.00199351, 0.00199351, 0.00199351, 0.00199351],
              [0.04521008, 0.04521008, 0.04521008, 0.04521008, 0.04521008, 0.04521008, 0.04521008, 0.04521008]],
             dtype=np.float32)
)

default_fc2_weight_adamw_v = (
    np.array([[35.217834, 42.283375, 26.52298, 21.510029]], dtype=np.float32)
)


class MockTransformerConfig:
    """Mock transformer config for testing Muon optimizer."""
    def __init__(self):
        self.multi_latent_attention = True
        self.tensor_model_parallel_size = 1
        self.data_parallel_size = 1


class MockModel:
    """
    Mock model class that provides required interfaces for Muon optimizer.
    This simulates the model interface that Muon optimizer expects.
    """
    def __init__(self):
        self.config = MockTransformerConfig()

    def get_gpt_transformer_config(self):
        """Return transformer config."""
        return self.config

    def make_model_muon_fns(self):
        """Return muon split and merge functions."""
        def muon_split_fn(param_name, tensor):  # pylint: disable=unused-argument
            """Split function - returns tensor as list."""
            return [tensor]

        def muon_merge_fn(param_name, tensor_list):  # pylint: disable=unused-argument
            """Merge function - returns first tensor."""
            return tensor_list[0]

        return muon_split_fn, muon_merge_fn

    def get_param_layer_indices(self, params):
        """Return layer indices for parameters."""
        return {p.name: 0 for p in params}

    def get_muon_filter(self):
        """Return filter function to determine which params use Muon."""
        def muon_filter(param):
            # Apply Muon to weight parameters with 2D shape (not bias)
            return len(param.shape) == 2 and 'bias' not in param.name
        return muon_filter

    def get_tp_dims(self, params):
        """Return tensor parallel dimensions."""
        return tuple(-1 for _ in params)

    def get_op_groups_info(self, params, op):  # pylint: disable=unused-argument
        """Return optimizer parallel group info."""
        ops = tuple(1 for _ in params)
        op_groups = tuple("" for _ in params)
        return ops, op_groups


def build_muon_network(net, mock_model, learning_rate=0.02):
    """
    Build network with Muon optimizer for testing.

    Args:
        net: The network to train
        mock_model: Mock model providing Muon interface
        learning_rate: Learning rate for optimizer

    Returns:
        tuple: (losses, optimizer)
    """

    loss_fn = nn.L1Loss(reduction='mean')
    networkwithloss = NetWithLoss(net, loss_fn)
    networkwithloss.set_train()

    params = networkwithloss.trainable_params()

    # Create Muon optimizer
    optimizer = Muon(
        params=params,
        learning_rate=learning_rate,
        weight_decay=0.1,
        matched_adamw_rms=0.2,
        momentum=0.95,
        nesterov=True,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        model=mock_model,
    )

    trainonestepcell = mindspore.nn.TrainOneStepCell(networkwithloss, optimizer)

    losses = []
    data, label = make_fake_data()
    for i in range(20):
        loss = trainonestepcell(data[i], label[i])
        losses.append(loss.asnumpy())

    return np.array(losses), optimizer
