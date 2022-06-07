# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Some basic usage functions """
from dataclasses import dataclass
import os
import time
import numpy as np

from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter, ParameterTuple

from mindspore import nn, dtype


@dataclass
class ModelSize:
    """The model statistics for print model size"""
    parameters: int = 0
    size: int = 0
    size_unit: str = 'B'

    def find_unit(self):
        candidate_list = ['KB', 'MB', "GB", "TB"]
        for item in candidate_list:
            if self.size // 1024 > 0:
                self.size /= 1024
                self.size_unit = item
            else:
                break


def print_mode_size(net: nn.Cell):
    """Print the number of parameters and its size"""
    net_size = ModelSize()
    trainable_net_size = ModelSize()
    model_size = {f"{dtype.float32}": 4, f"{dtype.float16}": 2, f"{dtype.float64}": 8}
    for _, param in net.parameters_and_names():
        n = np.prod(param.shape)
        size = n * model_size[f"{param.dtype}"]
        net_size.parameters += n
        net_size.size += size
        if param.requires_grad:
            trainable_net_size.parameters += n
            trainable_net_size.size += size
    net_size.find_unit()
    trainable_net_size.find_unit()
    print(f"The statistics of the net:")
    print(f"{'The number of parameters':<40}:{net_size.parameters:.1E},\t "
          f"{'Model size':}:{net_size.size:.1E} {net_size.size_unit}", flush=True)
    print(f"{'The number of trainable Parameters':<40}:{trainable_net_size.parameters:.1E},\t "
          f"{'Model size':<2}:{trainable_net_size.size:.1E} {net_size.size_unit}", flush=True)


def clone_state(parameter_tuple, prefix, init):
    r"""
        Clone the float32 copies of the parameter
        parameter_tuple: ParameterTuple. The parameters of the network
        prefix: str. The prefix name of the parameters
        init: str. The initialization method
    """
    new = []
    for old_param in parameter_tuple:
        param_init = init
        if init is None:
            param_init = old_param.init
        new_state = Parameter(initializer(param_init, shape=old_param.shape, dtype=mstype.float32))
        new_state.param_info = old_param.param_info.clone()
        new_state.is_init = False
        new_state.is_param_ps = old_param.is_param_ps
        new_state.init_in_server = old_param.init_in_server
        new_state.cache_enable = old_param.cache_enable
        new_state.requires_aggr = old_param.requires_aggr
        if old_param.cache_shape:
            new_state.cache_shape = old_param.cache_shape
        new_state.name = prefix + '.' + new_state.name
        new.append(new_state)
    return ParameterTuple(new)


def download_data(src_data_url, tgt_data_path, rank):
    """
        Download the dataset from the obs.
        src_data_url (Str): should be the dataset path in the obs
        tgt_data_path (Str): the local dataset path
        rank (Int): the current rank id

    """
    cache_url = tgt_data_path
    tmp_path = '/tmp'
    if rank % 8 == 0:
        import moxing as mox
        print("Modify the time out from 300 to 30000")
        print("begin download dataset", flush=True)

        if not os.path.exists(cache_url):
            os.makedirs(cache_url, exist_ok=True)
        mox.file.copy_parallel(src_url=src_data_url,
                               dst_url=cache_url)
        print("Dataset download succeed!", flush=True)

        f = open("%s/install.txt" % (tmp_path), 'w')
        f.close()
    # stop
    while not os.path.exists("%s/install.txt" % (tmp_path)):
        time.sleep(1)
