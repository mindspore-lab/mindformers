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
"""ModelMixin for train models"""
import numpy as np


class TrainModelMixin:
    """General interfaces for train models."""
    def concat_qkv_weight_megatron(self, wq_keys, wk_keys, wv_keys, qkv_weight_dict, condition, ms_weight_dict,
                                   head_dim, n_kv_heads, num_attention_heads):
        """
        Concat qkv weight from dicts like megatron format.

        Args:
            wq_keys: Query weight name.
            wk_keys: Key weight name.
            wv_keys: Value weight name.
            qkv_weight_dict: Query, key, value weight dict.
            condition: Condition to manager context.
            ms_weight_dict: Converted weight dict.
            head_dim: Projection weights dimension in multi-head attention.
            n_kv_heads: Number of query groups for group query attention.
            num_attention_heads: Number of transformer attention heads.
        """
        n_rep = num_attention_heads // n_kv_heads

        # Pop extra weight to shared dict if there is no corresponding weight for concat in the target dict
        for wk_key in wk_keys:
            wq_key = wk_key.replace('linear_k', 'linear_q')
            if wq_key not in wq_keys:
                with condition:
                    qkv_weight_dict[wk_key] = ms_weight_dict.pop(wk_key)  # add extra weight to shared dict
                    condition.notify_all()

        for wv_key in wv_keys:
            wq_key = wv_key.replace('linear_v', 'linear_q')
            if wq_key not in wq_keys:
                with condition:
                    qkv_weight_dict[wv_key] = ms_weight_dict.pop(wv_key)  # add extra weight to shared dict
                    condition.notify_all()

        # Concat qkv
        for wq_key in wq_keys:
            wk_key = wq_key.replace('linear_q', 'linear_k')
            wv_key = wq_key.replace('linear_q', 'linear_v')

            wq_value = ms_weight_dict.pop(wq_key)
            wk_value = ms_weight_dict.pop(wk_key, None)
            wv_value = ms_weight_dict.pop(wv_key, None)

            # Get missing weight from shared dict
            if wk_value is None:
                with condition:
                    condition.wait_for(lambda: wk_key in qkv_weight_dict.keys())
                    wk_value = qkv_weight_dict.pop(wk_key)

            if wv_value is None:
                with condition:
                    condition.wait_for(lambda: wv_key in qkv_weight_dict.keys())
                    wv_value = qkv_weight_dict.pop(wv_key)

            # Start to concat qkv weight
            w_qkv_key = wq_key.replace('linear_q', 'linear_qkv')

            # Reshape the q/k/v weight to 3d
            q_reshape = wq_value.reshape(n_kv_heads, n_rep * head_dim, -1)
            k_reshape = wk_value.reshape(n_kv_heads, head_dim, -1)
            v_reshape = wv_value.reshape(n_kv_heads, head_dim, -1)

            # Then concat them with column (axis 1)
            concat_qkv_weight = np.concatenate((q_reshape, k_reshape, v_reshape), axis=1)

            # Reshape the concated qkv weight to 2d
            w_qkv_value = concat_qkv_weight.reshape((n_rep + 2) * head_dim * n_kv_heads, -1)

            # update converted qkv weight to `ms_weight_dict`
            ms_weight_dict.update({w_qkv_key: w_qkv_value})

    def concat_ffn_weight_megatron(self, w1_keys, w3_keys, ffn_weight_dict, condition, ms_weight_dict, ffn_hidden_size):
        """
        Concat ffn weight from dicts.

        Args:
            w1_keys: FFN w1 weight name.
            w3_keys: FFN w3 weight name.
            ffn_weight_dict: FFN weight dict.
            condition: Condition to manager context.
            ms_weight_dict: Converted weight dict.
            ffn_hidden_size: Feed-Forward Network hidden size.

        """

        # Pop extra weight to shared dict if there is no corresponding weight for concat in the target dict
        for w3_key in w3_keys:
            w1_key = w3_key.replace('hidden', 'gating')
            if w1_key not in w1_keys:
                with condition:
                    ffn_weight_dict[w3_key] = ms_weight_dict.pop(w3_key)  # add extra weight to shared dict
                    condition.notify_all()

        # Concat ffn
        for w1_key in w1_keys:
            w3_key = w1_key.replace('gating', 'hidden')
            w1_value = ms_weight_dict.pop(w1_key)
            w3_value = ms_weight_dict.pop(w3_key, None)

            # Get missing weight from shared dict
            if w3_value is None:
                with condition:
                    condition.wait_for(lambda: w3_key in ffn_weight_dict.keys())
                    w3_value = ffn_weight_dict.pop(w3_key)

            w_gate_hidden_key = w1_key.replace('gating', 'linear_fc1')

            # Reshape w1 w3 to 3d
            gate_reshape = w1_value.reshape(ffn_hidden_size, 1, -1)
            hidden_reshape = w3_value.reshape(ffn_hidden_size, 1, -1)

            # Concat w1 and w3
            concat_ffn_weight = np.concatenate((gate_reshape, hidden_reshape), axis=1)

            # Reshape the concated ffn weight to 2d
            w_gate_hidden_value = concat_ffn_weight.reshape(ffn_hidden_size * 2, -1)

            # Update converted ffn weight to `ms_weight_dict`
            ms_weight_dict.update({w_gate_hidden_key: w_gate_hidden_value})

    def concat_qkv_weight_infer(self, wq_keys, wk_keys, wv_keys, qkv_weight_dict, condition, ms_weight_dict):
        r"""
        concat qkv weight from dicts.

        Args:
            wq_keys: Query weight name.
            wk_keys: Key weight name.
            wv_keys: Value weight name.
            qkv_weight_dict: Query, key, value weight dict.
            condition: Condition to manager context.
            ms_weight_dict: Converted weight dict.
        """
        # pop extra weight to shared dict if there is no corresponding weight for concat in the target dict
        for wk_key in wk_keys:
            wq_key = wk_key.replace('linear_k', 'linear_q')
            if wq_key not in wq_keys:
                with condition:
                    qkv_weight_dict[wk_key] = ms_weight_dict.pop(wk_key)  # add extra weight to shared dict
                    condition.notify_all()

        for wv_key in wv_keys:
            wq_key = wv_key.replace('linear_v', 'linear_q')
            if wq_key not in wq_keys:
                with condition:
                    qkv_weight_dict[wv_key] = ms_weight_dict.pop(wv_key)  # add extra weight to shared dict
                    condition.notify_all()

        # concat qkv
        for wq_key in wq_keys:
            wk_key = wq_key.replace('linear_q', 'linear_k')
            wv_key = wq_key.replace('linear_q', 'linear_v')

            wq_value = ms_weight_dict.pop(wq_key)
            wk_value = ms_weight_dict.pop(wk_key, None)
            wv_value = ms_weight_dict.pop(wv_key, None)

            # get missing weight from shared dict
            if wk_value is None:
                with condition:
                    condition.wait_for(lambda: wk_key in qkv_weight_dict.keys())
                    wk_value = qkv_weight_dict.pop(wk_key)

            if wv_value is None:
                with condition:
                    condition.wait_for(lambda: wv_key in qkv_weight_dict.keys())
                    wv_value = qkv_weight_dict.pop(wv_key)

            w_qkv_key = wq_key.replace('linear_q', 'linear_qkv')
            w_qkv_value = np.concatenate((wq_value, wk_value, wv_value), 0)

            # update converted qkv weight to `ms_weight_dict`
            ms_weight_dict.update({w_qkv_key: w_qkv_value})

    def concat_ffn_weight_infer(self, w1_keys, w3_keys, ffn_weight_dict, condition, ms_weight_dict):
        """
        concat ffn weight from dicts.

        Args:
            w1_keys: FFN w1 weight name.
            w3_keys: FFN w3 weight name.
            ffn_weight_dict: FFN weight dict.
            condition: Condition to manager context.
            ms_weight_dict: Converted weight dict.
        """

        # pop extra weight to shared dict if there is no corresponding weight for concat in the target dict
        for w3_key in w3_keys:
            w1_key = w3_key.replace('hidden', 'gating')
            if w1_key not in w1_keys:
                with condition:
                    ffn_weight_dict[w3_key] = ms_weight_dict.pop(w3_key)  # add extra weight to shared dict
                    condition.notify_all()

        # concat ffn
        for w1_key in w1_keys:
            w3_key = w1_key.replace('gating', 'hidden')
            w1_value = ms_weight_dict.pop(w1_key)
            w3_value = ms_weight_dict.pop(w3_key, None)

            # get missing weight from shared dict
            if w3_value is None:
                with condition:
                    condition.wait_for(lambda: w3_key in ffn_weight_dict.keys())
                    w3_value = ffn_weight_dict.pop(w3_key)

            w_gate_hidden_key = w1_key.replace('gating', 'linear_fc1')
            w_gate_hidden_value = np.concatenate((w1_value, w3_value), 0)

            # update converted ffn weight to `ms_weight_dict`
            ms_weight_dict.update({w_gate_hidden_key: w_gate_hidden_value})

    def get_model_parameters(self):
        """Get current rank trainable parameters in model ."""
        return self.model.get_model_parameters()
