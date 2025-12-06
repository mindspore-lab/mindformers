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

from mindspore import Tensor
import mindspore.common.dtype as mstype
import numpy as np

from mindformers.tools.logger import logger
from mindformers.core.context.build_context import is_legacy_model
from mindformers.parallel_core.transformer_config_utils import convert_to_transformer_config


class ModelMixin:
    """
    A few utilities for `mindspore.nn.Cell`, to be used as a mixin.
    """

    concat_mapping = [
        ('.linear_q.', '.linear_qkv.'),
        ('.linear_k.', '.linear_qkv.'),
        ('.linear_v.', '.linear_qkv.'),
        ('.mlp.gating.', '.mlp.linear_fc1.'),
        ('.mlp.hidden.', '.mlp.linear_fc1.'),
    ]

    def __init__(self):
        self.transformer_config = None

    def convert_concat_name(self, weight_name):
        r"""
        convert HuggingFace weight name to MindFormers weight name.

        Args:
            weight_name: huggingface weight names.

        Returns:
            weight_name: converted weight names.

        """
        if is_legacy_model():
            raise RuntimeError(f"{self.__class__.__name__} does not implemented convert_name method.")
        for split_name, concat_name in self.concat_mapping:
            if split_name in weight_name:
                weight_name = weight_name.replace(split_name, concat_name)
        return weight_name

    def convert_name(self, weight_name):
        r"""
        convert HuggingFace weight name to MindFormers weight name.

        Args:
            weight_name: huggingface weight names.

        Returns:
            weight_name: converted weight names.

        """
        if is_legacy_model():
            raise RuntimeError(f"{self.__class__.__name__} does not implemented convert_name method.")
        for hf_name, mcore_name in self.weight_mapping:
            if hf_name in weight_name:
                weight_name = weight_name.replace(hf_name, mcore_name)
        return weight_name

    def set_dynamic_inputs(self, **kwargs):
        """
        Compile static graphs into dynamic shapes
        """

        raise RuntimeError(
            "A model class needs to define a `set_dynamic_inputs`"
            " method in order to use `model.set_inputs()`."
        )

    def convert_to_transformer_config(self, model_config, is_mla_model: bool = False, additional_map: dict = None,
                                      not_convert_whitelist: set = None):
        self.transformer_config = convert_to_transformer_config(
            model_config, is_mla_model, additional_map, not_convert_whitelist
        )
        return self.transformer_config

    def get_gpt_transformer_config(self):
        """
        Get the transformer config for GPT model
        """
        if self.transformer_config is None:
            raise ValueError("Please call `convert_to_transformer_config` in Model before get gpt transformer config.")
        return self.transformer_config

    def is_mtp_model(self):
        """Check whether the model is a multi-token prediction model."""
        mtp_num_layers = self.get_gpt_transformer_config().mtp_num_layers
        return bool(mtp_num_layers and mtp_num_layers > 0)

    def is_moe_model(self):
        """Check whether the model is a moe model."""
        num_moe_experts = self.get_gpt_transformer_config().num_moe_experts
        return bool(num_moe_experts and num_moe_experts > 0)

    def get_gpt_model(self):
        """
        Obtain the GPT model instance.
        """
        if not hasattr(self, 'model'):
            raise RuntimeError("Mcore model definition should use the fixed paradigm: "
                               "self.model = GPTModel(*args, **kwargs) definition. "
                               "Currently, this attribute cannot be correctly recognized. "
                               "Please modify the GPTModel definition method.")
        return getattr(self, 'model')

    @classmethod
    def convert_weight_dict(cls, source_dict, **kwargs):
        """convert HuggingFace weight dict to MindFormers weight dict"""
        raise RuntimeError(f"{cls.__name__} does not implemented convert_weight_dict method.")

    @classmethod
    def convert_map_dict(cls, source_dict, **kwargs):
        """convert HuggingFace map dict to MindFormers map dict"""
        raise RuntimeError(f"{cls.__name__} does not implemented convert_map_dict method.")

    @classmethod
    def obtain_name_map(cls, load_checkpoint_files):
        """obtain HuggingFace safetensor name_map dict to MindFormers """
        raise RuntimeError(f"{cls.__name__} does not implemented obtain_name_map method.")

    @classmethod
    def obtain_qkv_ffn_concat_keys(cls):
        """
        Obtain key list generated during weight concatenation of qkv/ffn concat operation.
        For example:
        When qkv/ffn concat weight conversion operation is performed on llama model,
        new keys containing "w_qkv" and "w_gate_hidden" are generated.

        Returns:
            key_list (list): key word list of concat weights.
        """
        logger.info(f"{cls.__name__} does not support qkv concat check, skipping...")


class TrainModelMixin:
    """General interfaces for train models."""

    is_train_model = True

    def concat_qkv_contiguous(self, q_value, k_value, v_value, q_name):
        """
        Concat the Q/K/V weight in contiguous format:
            [Q_weights, K_weights, V_weights].
        """
        qkv_name = q_name.replace('linear_q', 'linear_qkv')
        qkv_value = np.concatenate((q_value, k_value, v_value), 0)

        # return converted qkv weight
        return {qkv_name: qkv_value}

    def concat_qkv_interleaved(self, q_value, k_value, v_value, q_name,
                               head_dim, n_kv_heads, num_attention_heads):
        """
        Concat the Q/K/V weight in interleaved format:
            [Q_head0, K_head0, V_head0, Q_head1, ...].
        """
        n_rep = num_attention_heads // n_kv_heads

        # Start to concat qkv weight
        qkv_name = q_name.replace('linear_q', 'linear_qkv')

        # Reshape the q/k/v weight to 3d
        q_reshape = q_value.reshape(n_kv_heads, n_rep * head_dim, -1)
        k_reshape = k_value.reshape(n_kv_heads, head_dim, -1)
        v_reshape = v_value.reshape(n_kv_heads, head_dim, -1)

        # Then concat them with column (axis 1)
        concat_qkv_weight = np.concatenate((q_reshape, k_reshape, v_reshape), axis=1)

        # Reshape the concated qkv weight to 2d
        qkv_value = concat_qkv_weight.reshape((n_rep + 2) * head_dim * n_kv_heads, -1)

        # return converted qkv weight
        return {qkv_name: qkv_value}

    def concat_linear_fc1_contiguous(self, gate_value, up_value, gate_name):
        """
        Concat the gate/up weight in contiguous format:
            [Gate_weights, Hidden_weights].
        """
        linear_fc1_key = gate_name.replace('gating', 'linear_fc1')
        linear_fc1_value = np.concatenate((gate_value, up_value), 0)

        # return converted ffn weight
        return {linear_fc1_key: linear_fc1_value}

    def concat_linear_fc1_interleaved(self, gate_value, up_value, gate_name, ffn_hidden_size):
        """
        Concat the gate/up weight in interleaved format:
            [Gate_weights[0], Hidden_weights[0], Gate_weights[1], Hidden_weights[1], ...].
        """
        linear_fc1_key = gate_name.replace('gating', 'linear_fc1')

        # Reshape gate/up to 3d
        gate_reshape = gate_value.reshape(ffn_hidden_size, 1, -1)
        hidden_reshape = up_value.reshape(ffn_hidden_size, 1, -1)

        # Concat gate and up
        linear_fc1_value = np.concatenate((gate_reshape, hidden_reshape), axis=1)

        # Reshape the concated linear_fc1 weight to 2d
        linear_fc1_value = linear_fc1_value.reshape(ffn_hidden_size * 2, -1)

        # return converted ffn weight
        return {linear_fc1_key: linear_fc1_value}

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

    def concat_expert_weight(self, w2_keys, expert_weight_dict, condition, ms_weight_dict, num_layers, num_experts):
        """
        Concat ffn weight from dicts.

        Args:
            w2_keys: FFN w2 in experts weight name.
            expert_weight_dict: Experts weight dict.
            condition: Condition to manager context.
            ms_weight_dict: Converted weight dict.
            num_layers: Number of transformer layers in a transformer block.
            num_experts: Number of experts to use for MoE layer.

        """
        base_prefix = "decoder.layers._.mlp.experts."

        # Traverse and process the expert layer weights for each sub-process by layer.
        for layer in range(0, num_layers):
            cur_layer_prefix = base_prefix.replace('_', str(layer))

            linear_fc2_first_key = cur_layer_prefix + '0.linear_fc2.weight'
            linear_fc1_first_key = linear_fc2_first_key.replace('linear_fc2', 'linear_fc1')
            concat_fc1 = True
            concat_fc2 = True

            # Check whether the 'experts.0.linear_fc1.weight' is used in this process.
            # If not, add it to the shared dict 'expert_weight_dict'.
            if linear_fc1_first_key not in ms_weight_dict.keys():
                concat_fc1 = False
                for expert_idx in range(0, num_experts):
                    linear_fc1_key = cur_layer_prefix + str(expert_idx) + '.linear_fc1.weight'
                    if linear_fc1_key in ms_weight_dict.keys():
                        with condition:
                            expert_weight_dict[linear_fc1_key] = ms_weight_dict.pop(linear_fc1_key)
                            condition.notify_all()

            # Check whether the 'experts.0.linear_fc2.weight' is used in this process.
            # If not, add it to the shared dict 'expert_weight_dict'.
            if linear_fc2_first_key not in w2_keys:
                concat_fc2 = False
                for expert_idx in range(0, num_experts):
                    linear_fc2_key = cur_layer_prefix + str(expert_idx) + '.linear_fc2.weight'
                    if linear_fc2_key in ms_weight_dict.keys():
                        with condition:
                            expert_weight_dict[linear_fc2_key] = ms_weight_dict.pop(linear_fc2_key)
                            condition.notify_all()

            # If the subprocess does not need to concat the expert weights of this layer,
            # it will skip checking the next layer directly.
            if not concat_fc1 and not concat_fc2:
                continue

            cur_layer_linear_fc1_weights_dict = []
            cur_layer_linear_fc2_weights_dict = []

            if concat_fc1:
                # Get all fc1 weights of this layer in order starting from '0'.
                for expert_idx in range(0, num_experts):
                    linear_fc1_key = cur_layer_prefix + str(expert_idx) + '.linear_fc1.weight'
                    fc1_value = ms_weight_dict.pop(linear_fc1_key, None)
                    if fc1_value is None:
                        with condition:
                            # If the current process cannot obtain the weight, try to obtain it in the shared dict.
                            condition.wait_for(lambda: linear_fc1_key in expert_weight_dict.keys())
                            fc1_value = expert_weight_dict.pop(linear_fc1_key)
                    cur_layer_linear_fc1_weights_dict.append(fc1_value)

                # Check whether all fc1 expert weights in this layer have been obtained,
                # otherwise splicing cannot be performed.
                if len(cur_layer_linear_fc1_weights_dict) == num_experts:
                    # Stack each expert up weight into 3d.
                    cur_layer_linear_fc1_weight = np.stack(cur_layer_linear_fc1_weights_dict, axis=0)
                    # And transpose their shape to (num_experts, hidden_size, 2 * moe_ffn_hidden_size)
                    cur_layer_linear_fc1_weight = cur_layer_linear_fc1_weight.transpose(0, 2, 1)
                    # Then reshape them into 2d (num_experts * hidden_size, 2 * moe_ffn_hidden_size)
                    cur_layer_linear_fc1_weight = cur_layer_linear_fc1_weight.reshape(
                        num_experts * self.transformer_config.hidden_size, -1
                    )
                    # Replace the key of this weight.
                    cur_layer_linear_fc1_key = cur_layer_prefix + 'weight1'
                    # Update the weight dictionary handled by the subprocess.
                    ms_weight_dict.update({cur_layer_linear_fc1_key: cur_layer_linear_fc1_weight})
                else:
                    raise ValueError(f"the length of cur_layer_linear_fc1_weights_dict is "
                                     f"{len(cur_layer_linear_fc1_weights_dict)}, can't stack them.")

            if concat_fc2:
                # Get all fc2 weights of this layer in order starting from '0'.
                for expert_idx in range(0, num_experts):
                    linear_fc2_key = cur_layer_prefix + str(expert_idx) + '.linear_fc2.weight'
                    fc2_value = ms_weight_dict.pop(linear_fc2_key, None)
                    if fc2_value is None:
                        with condition:
                            # If the current process cannot obtain the weight, try to obtain it in the shared dict.
                            condition.wait_for(lambda: linear_fc2_key in expert_weight_dict.keys())
                            fc2_value = expert_weight_dict.pop(linear_fc2_key)
                    cur_layer_linear_fc2_weights_dict.append(fc2_value)

                # Check whether all fc2 expert weights in this layer have been obtained,
                # otherwise splicing cannot be performed.
                if len(cur_layer_linear_fc2_weights_dict) == num_experts:
                    # Stack each expert up weight into 3d.
                    cur_layer_linear_fc2_weight = np.stack(cur_layer_linear_fc2_weights_dict, axis=0)
                    # And transpose their shape to (num_experts, moe_ffn_hidden_size, hidden_size)
                    cur_layer_linear_fc2_weight = cur_layer_linear_fc2_weight.transpose(0, 2, 1)
                    # Then reshape them into 2d (num_experts * moe_ffn_hidden_size, hidden_size)
                    cur_layer_linear_fc2_weight = cur_layer_linear_fc2_weight.reshape(
                        num_experts * self.transformer_config.moe_ffn_hidden_size, -1
                    )
                    # Replace the key of this weight.
                    cur_layer_linear_fc2_key = cur_layer_prefix + 'weight2'
                    # Update the weight dictionary handled by the subprocess.
                    ms_weight_dict.update({cur_layer_linear_fc2_key: cur_layer_linear_fc2_weight})
                else:
                    raise ValueError(f"the length of cur_layer_linear_fc2_weights_dict is "
                                     f"{len(cur_layer_linear_fc2_weights_dict)}, can't stack them.")

    def check_and_get_model(self):
        """Check and get GPT model instance."""
        if not hasattr(self, 'model'):
            raise RuntimeError("Mcore model definition should use the fixed paradigm: "
                               "self.model = GPTModel(*args, **kwargs) definition. "
                               "Currently, this attribute cannot be correctly recognized. "
                               "Please modify the GPTModel definition method.")
        return getattr(self, 'model')

    def get_model_parameters(self):
        """Get current rank trainable parameters in model ."""
        model = self.check_and_get_model()
        return model.get_model_parameters()

    def get_max_attention_logit(self):
        """Get max attention logit values from the model."""
        model = self.check_and_get_model()
        return model.get_max_attention_logit()

    def make_model_muon_fns(self):
        """Make model muon functions."""
        model = self.check_and_get_model()
        return model.make_model_muon_fns()

    def get_muon_filter(self):
        """Get muon filter."""
        model = self.check_and_get_model()
        return model.get_muon_filter()

    def get_tp_dims(self, parameters):
        """Get tensor parallel dimensions for parameters."""
        model = self.check_and_get_model()
        return model.get_tp_dims(parameters)

    def get_op_groups_info(self, parameters, op_size):
        """Get operation groups information for parameters."""
        model = self.check_and_get_model()
        return model.get_op_groups_info(parameters, op_size)

    def get_parallel_config_for_muon(self):
        """Get parallel configuration for Muon optimizer."""
        model = self.check_and_get_model()
        return model.get_parallel_config_for_muon()

    def get_param_layer_indices(self, parameters):
        """Get layer indices for parameters."""
        model = self.check_and_get_model()
        return model.get_param_layer_indices(parameters)

    def apply_qk_clip_scaling(self, parameters, param_names, param_layers,
                              logit_threshold, split_fn, merge_fn):
        """Apply QK clip scaling to parameters."""
        model = self.check_and_get_model()
        return model.apply_qk_clip_scaling(
            parameters, param_names, param_layers,
            logit_threshold, split_fn, merge_fn
        )

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        return Tensor(input_ids, mstype.int32), None, None, None, None, None, None, None, None
