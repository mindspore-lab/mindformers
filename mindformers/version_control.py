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
"""MindSpore Version Control"""
import os
import mindspore as ms
from mindspore import nn
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from .tools.utils import is_version_ge
from .tools.logger import logger

def get_cell_reuse():
    """Cell reuse decorator."""
    def decorator(func):
        if os.getenv("MS_DEV_CELL_REUSE", "0") == "0" or not is_version_ge(ms.__version__, "2.1.0"):
            return func
        logger.info("Enable cell use mode at %s.", func.__class__.__name__)
        from mindspore._extends import cell_attr_register
        return cell_attr_register()(func)
    return decorator

def get_dropout(dropout_prob):
    if is_version_ge(ms.__version__, '1.11.0'):
        dropout = nn.Dropout(p=dropout_prob)
    else:
        dropout = nn.Dropout(keep_prob=1 - dropout_prob)
    return dropout

def get_tril():
    if is_version_ge(ms.__version__, '1.11.0'):
        tril = P.Tril()
    else:
        tril = nn.Tril()
    return tril

def get_norm():
    """return ops.norm"""
    # pylint: disable=C0103
    # pylint: disable=W0622
    def tensor_norm1(A, ord=None, dim=None, keepdim=False, dtype=None):
        return F.norm(A, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)

    # pylint: disable=C0103
    # pylint: disable=W0622
    def tensor_norm2(A, ord=2, dim=None, keepdim=False, dtype=None):
        if dtype is not None:
            logger.warning("The 'dtype' is not available when mindspore version < '1.11.0'")
        if not isinstance(ord, int):
            raise TypeError("The type of 'ord' should be int when mindspore version < '1.11.0'")
        return F.norm(A, dim, p=ord, keep_dims=keepdim)

    if is_version_ge(ms.__version__, '1.11.0'):
        return tensor_norm1
    return tensor_norm2

def get_dataset_map(dataset, operations, input_columns=None, output_columns=None, num_parallel_workers=None, **kwargs):
    if is_version_ge(ms.__version__, "1.11.0"):
        return dataset.map(operations,
                           input_columns=input_columns,
                           output_columns=output_columns,
                           num_parallel_workers=num_parallel_workers,
                           **kwargs)
    return dataset.map(operations,
                       input_columns=input_columns,
                       output_columns=output_columns,
                       column_order=output_columns,
                       num_parallel_workers=num_parallel_workers,
                       python_multiprocessing=kwargs.get('python_multiprocessing', False),
                       max_rowsize=kwargs.get('max_rowsize', 16),
                       cache=kwargs.get('cache', None),
                       callbacks=kwargs.get('callbacks', None),
                       offload=kwargs.get('offload', None))

def get_identity():
    if is_version_ge(ms.__version__, "1.11.0"):
        return nn.Identity()
    return F.identity
