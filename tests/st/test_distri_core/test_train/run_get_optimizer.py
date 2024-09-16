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
"""run get optimizer"""
import argparse

import mindspore as ms
import mindspore.nn as nn
from mindspore.communication import init

from mindformers.experimental.parallel_core.pynative.parallel_state import (
    initialize_model_parallel,
)

from mindformers.experimental.parallel_core.pynative.config import (
    init_configs_from_yaml
)

from mindformers.experimental.parallel_core.pynative.optimizer import get_optimizer


class TestNet(nn.Cell):
    """ TestNet. """
    def __init__(self):
        super(TestNet, self).__init__()
        self.linear = nn.Dense(128, 128)
        self.linear_weight_decay = nn.Dense(128, 128)

    def construct(self, x):
        x = self.linear(x)
        x = self.linear_weight_decay(x)
        return x


def set_weight_decay(params, weight_decay=1e-1):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest

    Args:
        params (list[Parameter]): List of parameters to apply weight decay to.

    Returns:
        list: A list of dictionaries specifying the parameter groups and their respective weight decay coefficients.
    """
    decay_params = list(filter(lambda x: "weight_decay" in x.name.lower(), params))
    other_params = list(filter(lambda x: "weight_decay" not in x.name.lower(), params))
    group_params = [
        {"order_params": params},
    ]
    if decay_params:
        group_params.append({"params": decay_params, "weight_decay": weight_decay})
    if other_params:
        group_params.append({"params": other_params, "weight_decay": 0.0})
    return group_params


def run_get_optimizer(config_path, group_params=False):
    """run get optimizer"""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

    all_config = init_configs_from_yaml(config_path)
    optimizer_config = all_config.optimizer_config
    parallel_config = all_config.parallel_config

    init()
    initialize_model_parallel(
        tensor_model_parallel_size=parallel_config.tensor_parallel,
    )

    net = TestNet()
    if group_params:
        params = set_weight_decay(net.trainable_params())
    else:
        params = net.trainable_params()
    optimizer = get_optimizer(optimizer_config, params, net)
    print(optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DistriCore get optimizer')
    parser.add_argument('--config_path', type=str, default='config.yaml', help='config path')
    parser.add_argument('--group_params', type=bool, default=False, help='group params')
    args_opt = parser.parse_args()
    run_get_optimizer(args_opt.config_path, args_opt.group_params)
