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
""" Script used for converting checkpoint to bfloat16 format. """
import os
import argparse

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.ops.operations import Cast

ms.set_context(device_target='CPU')
cpu_cast = Cast().set_device('CPU')

def convert_dtype(dtype_str):
    """Convert string to mindspore datatype.
    Args:
        dtype_str (str): String format data type argument.

    Raises:
        KeyError: Input string invalid.

    Returns:
        mstype: MindSpore datatype.
    """
    if dtype_str == "float16":
        return mstype.float16
    if dtype_str == "bfloat16":
        return mstype.bfloat16
    if dtype_str == "float32":
        return mstype.float32
    raise KeyError(f"Supported datatype keywords include:"
                   f"[float16, float32, bfloat16], but get {dtype_str}")

def convert(args):
    """Convert src checkpoint to bf16 format checkpoint.
    Args:
        args: Arguments.

    Raises:
        FileExistsError: Invalid src checkpoint path.
        FileExistsError: Invalid dst checkpoint path.
    """
    if not os.path.isfile(args.src_ckpt):
        raise FileExistsError(f'Invalid src checkpoint path {args.src_ckpt}.')
    param_dict = ms.load_checkpoint(args.src_ckpt)
    param_dict_converted = {}

    target_dtype = convert_dtype(args.dtype)
    for k, v in param_dict.items():
        if 'slice_sparse_mask' in k:
            continue
        if not args.embed_bf16 and 'embedding' in k:
            new_val = ms.Parameter(ms.Tensor(cpu_cast(v.data, mstype.float32).asnumpy()), name=k)
        elif not args.layernorm_bf16 and 'norm' in k:
            new_val = ms.Parameter(ms.Tensor(cpu_cast(v.data, mstype.float32).asnumpy()), name=k)
        else:
            new_val = ms.Parameter(ms.Tensor(cpu_cast(v.data, target_dtype).asnumpy()), name=k)
        param_dict_converted[k] = new_val

    if os.path.isdir(args.dst_ckpt):
        ms.save_checkpoint(param_dict_converted, os.path.join(args.dst_ckpt, 'checkpoint.ckpt'))
    else:
        ms.save_checkpoint(param_dict_converted, args.dst_ckpt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ckpt', required=True, type=str,
                        help='source checkpoint file path.')
    parser.add_argument('--dst_ckpt', required=True, type=str,
                        help='converted chekpoint save path.')
    parser.add_argument('--dtype', default='bfloat16', type=str,
                        help='converted datatype.')
    parser.add_argument('--embed_bf16', action='store_true',
                        help='embedding table use bfloat16.')
    parser.add_argument('--layernorm_bf16', action='store_true',
                        help='layernorm parameters use bfloat16.')
    _args = parser.parse_args()

    convert(_args)
