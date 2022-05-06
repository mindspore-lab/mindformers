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
