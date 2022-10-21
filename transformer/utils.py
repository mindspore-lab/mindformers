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


def _mapper_string_to_bool(argument):
    """Mapping the string true or false to bool value"""
    if argument in ['False', 'false']:
        return False
    if argument in ['True', 'true']:
        return True
    return argument


def _convert_dtype_class(key):
    """maps the mstype.float32 to real type. If found, return the target dtype, else return itself."""
    mapper = {'mstype.float32': mstype.float32, 'mstype.float16': mstype.float16,
              'fp32': mstype.float32, 'fp16': mstype.float16, 'None': None}
    return mapper.get(key, key)


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


def get_newest_ckpt(checkpoint_dir, prefix):
    """
    Find the newest ckpt path.
    """
    files = os.listdir(checkpoint_dir)
    max_time = 0
    newest_checkpoint_path = ""
    for filename in files:
        if filename.startswith(prefix) and filename.endswith(".ckpt"):
            full_path = os.path.join(checkpoint_dir, filename)
            mtime = os.path.getmtime(full_path)
            if mtime > max_time:
                max_time = mtime
                newest_checkpoint_path = full_path
    print("Find the newest checkpoint: ", newest_checkpoint_path)
    return newest_checkpoint_path


def print_model_size(net, logger):
    """Print the number of parameters and its size"""
    net_size = ModelSize()
    trainable_net_size = ModelSize()
    model_size = {f"{mstype.float32}": 4, f"{mstype.float16}": 2, f"{mstype.float64}": 8}
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
    logger.info(f"The statistics of the net:")
    logger.info(f"{'The number of parameters':<40}:{net_size.parameters:.1E},\t "
                f"{'Model size':}:{net_size.size:.1E} {net_size.size_unit}")
    logger.info(f"{'The number of trainable Parameters':<40}:{trainable_net_size.parameters:.1E},\t "
                f"{'Model size':<2}:{trainable_net_size.size:.1E} {net_size.size_unit}")


def clone_state(parameter_tuple, prefix, init, forced_dtype=mstype.float32, is_follow=False):
    r"""
        Clone the parameters
        parameter_tuple: ParameterTuple. The parameters of the network
        prefix: str. The prefix name of the parameters
        init: str. The initialization method
        forced_dtype: mstype. The except the dtype to be cloned. If is_follow is True, forced_dtype will be ignored.
               Default: mstype.float32
        is_follow: bool. Is clone the parameters with the original dtype. If is_follow is True, the forced_dtype
               argument will be ignored. Default: False.
    """
    new = []
    for old_param in parameter_tuple:
        param_init = init
        if init is None:
            param_init = old_param.init
        cur_dtype = forced_dtype
        if is_follow:
            cur_dtype = old_param.dtype
        new_state = Parameter(initializer(param_init, shape=old_param.shape, dtype=cur_dtype))
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


def download_data(src_data_path, tgt_data_path, rank):
    """
        Download the dataset from the obs.
        src_data_path (Str): should be the dataset path in the obs
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
        mox.file.copy_parallel(src_url=src_data_path,
                               dst_url=cache_url)
        print("Dataset download succeed!", flush=True)

        f = open("%s/install.txt" % (tmp_path), 'w')
        f.close()
    # stop
    while not os.path.exists("%s/install.txt" % (tmp_path)):
        time.sleep(1)


def make_directory(path: str, logger):
    """Make directory."""
    if path is None or not isinstance(path, str) or path.strip() == "":
        logger.error("The path(%r) is invalid type.", path)
        raise TypeError("Input path is invalid type")

    # convert the relative paths
    path = os.path.realpath(path)
    logger.debug("The abs path is %r", path)

    # check the path is exist and write permissions?
    if os.path.exists(path):
        real_path = path
    else:
        # All exceptions need to be caught because create directory maybe have some limit(permissions)
        logger.debug("The directory(%s) doesn't exist, will create it", path)
        try:
            os.makedirs(path, exist_ok=True)
            real_path = path
        except PermissionError as e:
            logger.error("No write permission on the directory(%r), error = %r", path, e)
            raise TypeError("No write permission on the directory.")
    return real_path
