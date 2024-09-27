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
"""User defined functions for PANGU-Alpha."""

def decay_filter(x):
    return "norm" not in x.name.lower() and "bias" not in x.name.lower()

def set_weight_decay(params, weight_decay=1e-1):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest

    Args:
        params (list[Parameter]): List of parameters to apply weight decay to.

    Returns:
        list: A list of dictionaries specifying the parameter groups and their respective weight decay coefficients.
    """
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [
        {"order_params": params},
    ]
    if decay_params:
        group_params.append({"params": decay_params, "weight_decay": weight_decay})
    if other_params:
        group_params.append({"params": other_params, "weight_decay": 0.0})
    return group_params
