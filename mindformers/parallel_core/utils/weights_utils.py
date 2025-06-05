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
"""attention and qkv concat."""
from safetensors import safe_open
import numpy as np


def concat_qkv_weight(wq_keys, load_checkpoint, weights, mf_hf_weights):
    r"""
    concat qkv weight from dicts.

    Args:
        wq_keys: query weight name.
        load_checkpoint: the path of storing weights.
        weights: weight dict.
        mf_hf_weights: the key is the value of the mf weights and the value is the value of the hf weights.

    Returns:

    """
    target_dict = {}
    for wq_key in wq_keys:
        wk_key = wq_key.replace('linear_q', 'linear_k')
        wv_key = wq_key.replace('linear_q', 'linear_v')
        wq_value = weights.pop(wq_key)
        wk_value = weights.pop(wk_key, None)
        wv_value = weights.pop(wv_key, None)
        if isinstance(wq_value, str) and wq_value.endswith('safetensors'):
            with safe_open(f"{load_checkpoint}/{wq_value}", framework="np") as sf_file:
                wq_value = sf_file.get_tensor(mf_hf_weights.get(wq_key))
            with safe_open(f"{load_checkpoint}/{wk_value}", framework="np") as sf_file:
                wk_value = sf_file.get_tensor(mf_hf_weights.get(wk_key))
            with safe_open(f"{load_checkpoint}/{wv_value}", framework="np") as sf_file:
                wv_value = sf_file.get_tensor(mf_hf_weights.get(wv_key))
        w_qkv_key = wq_key.replace('linear_q', 'linear_qkv')
        w_qkv_value = np.concatenate((wq_value, wk_value, wv_value), 0)
        target_dict.update({w_qkv_key: w_qkv_value})
    return target_dict


def concat_ffn_weight(w1_keys, load_checkpoint, weights, mf_hf_weights):
    r"""
    concat ffn weight from dicts.

    Args:
        w1_keys: ffn w1 weight name.
        load_checkpoint: the path of storing weights.
        weights: weight dict.
        mf_hf_weights: the key is the value of the mf weights and the value is the value of the hf weights.

    Returns:

    """
    target_dict = {}
    for w1_key in w1_keys:
        w3_key = w1_key.replace('gating', 'linear_fc1')
        w1_value = weights.pop(w1_key)
        w3_value = weights.pop(w3_key, None)
        if isinstance(w1_value, str) and w1_value.endswith('safetensors'):
            with safe_open(f"{load_checkpoint}/{w1_value}", framework="np") as sf_file:
                w1_value = sf_file.get_tensor(mf_hf_weights.get(w1_key))
            with safe_open(f"{load_checkpoint}/{w3_value}", framework="np") as sf_file:
                w3_value = sf_file.get_tensor(mf_hf_weights.get(w3_key))
        w_gate_hidden_key = w1_key.replace('gating', 'linear_fc1')
        w_gate_hidden_value = np.concatenate((w1_value, w3_value), 0)
        target_dict.update({w_gate_hidden_key: w_gate_hidden_value})
    return target_dict
