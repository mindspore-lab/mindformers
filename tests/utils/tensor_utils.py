# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Convert tensor tool.
"""
import mindspore as ms


def to_numpy_list(data):
    """
    Recursively convert a nested structure of Tensors to NumPy ndarrays and flatten them into a list.
    Supports nested tuples and lists.
    """
    result = []
    if isinstance(data, (tuple, list)):
        for item in data:
            result.extend(to_numpy_list(item))
    elif isinstance(data, ms.Tensor):
        result.append(data.asnumpy())
    else:
        raise TypeError(f"Unsupported type: {type(data)}. Expected Tensor or nested structure.")
    return result
