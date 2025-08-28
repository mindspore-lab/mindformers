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
"""Qwen3 models' APIs."""
from mindspore import Tensor

from mindformers.tools.logger import logger
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.training_graph.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from mindformers.parallel_core.utils.model_mixin import TrainModelMixin
from mindformers.models.qwen3.utils import Qwen3PreTrainedModel
from .configuration_qwen3 import Qwen3Config


class TrainingQwen3ForCausalLM(Qwen3PreTrainedModel, TrainModelMixin):
    r"""
    Provide qwen2 model infer through network.

    Args:
        config (Qwen3Config): The config of qwen3 model.

    Returns:
        output: Tensor, the output of qwen3 decoderlayer

    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config, auto_prefix=False)
        config: TransformerConfig = self.convert_to_transformer_config(self.config)

        self.model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_local_spec(
                qk_layernorm=True,
                use_contiguous_weight_layout_attention=config.use_contiguous_weight_layout_attention,
                use_interleaved_weight_layout_mlp=config.use_interleaved_weight_layout_mlp
            ),
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_position_embeddings,
            position_embedding_type=config.position_embedding_type,
            rotary_base=self.config.rope_theta,
            share_embeddings_and_output_weights=self.config.tie_word_embeddings,
            post_process=self.config.post_process
        )

    def construct(
            self,
            input_ids: Tensor,
            position_ids: Tensor = None,
            attention_mask: Tensor = None,
            decoder_input: Tensor = None,
            labels: Tensor = None,
            extra_block_kwargs=None,
            prefix_keys_values=None,
            loss_mask=None,
            actual_seq_len=None
    ):
        """Qwen3 construct for training"""
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            extra_block_kwargs=extra_block_kwargs,
            prefix_keys_values=prefix_keys_values,
            loss_mask=loss_mask,
            actual_seq_len=actual_seq_len
        )

    def convert_weight_dict(self, source_dict, **kwargs):
        r"""
        convert HuggingFace weight dict to MindFormers weight dict.

        Args:
            source_dict: origin weight dict.
            kwargs: additional specific kwargs.

        Raises:
            ValueError: value error

        Returns:
            ms_weight_dict: converted weight dict.

        """
        qkv_concat = kwargs.get("model_config").qkv_concat

        use_contiguous_weight_layout_attention = self.transformer_config.use_contiguous_weight_layout_attention

        ms_weight_dict = {}
        # QKV weight keys
        wq_keys = []
        wk_keys = []
        wv_keys = []
        # FFN weight keys
        w1_keys = []
        w3_keys = []

        for k, v in source_dict.items():
            k = self.convert_name(k)
            ms_weight_dict.update({k: v})

            if qkv_concat:
                part = k.split('.')
                # Get Q/K/V Keys
                if part[-2] == 'linear_q':
                    wq_keys.append(k)
                if part[-2] == 'linear_k':
                    wk_keys.append(k)
                if part[-2] == 'linear_v':
                    wv_keys.append(k)
                # Get FFN Keys in MLP
                if part[-2] == 'gating':
                    w1_keys.append(k)
                if part[-2] == 'hidden':
                    w3_keys.append(k)

        if qkv_concat:
            qkv_dict = kwargs.get('qkv_dict', None)
            condition = kwargs.get('condition', None)

            if use_contiguous_weight_layout_attention:
                logger.info("Concat QKV and FFN weight in contiguous weight layout attention.")
                self.concat_qkv_weight_infer(wq_keys, wk_keys, wv_keys, qkv_dict, condition, ms_weight_dict)
                self.concat_ffn_weight_infer(w1_keys, w3_keys, qkv_dict, condition, ms_weight_dict)
            else:
                logger.info("Concat QKV and FFN weight without contiguous weight layout attention.")
                self.concat_qkv_weight_megatron(
                    wq_keys=wq_keys, wk_keys=wk_keys, wv_keys=wv_keys,
                    qkv_weight_dict=qkv_dict, condition=condition, ms_weight_dict=ms_weight_dict,
                    head_dim=self.transformer_config.kv_channels,
                    n_kv_heads=self.transformer_config.num_query_groups,
                    num_attention_heads=self.transformer_config.num_attention_heads
                )
                self.concat_ffn_weight_megatron(
                    w1_keys=w1_keys, w3_keys=w3_keys,
                    ffn_weight_dict=qkv_dict, condition=condition, ms_weight_dict=ms_weight_dict,
                    ffn_hidden_size=self.transformer_config.ffn_hidden_size
                )

        return ms_weight_dict

    def convert_map_dict(self, hf_name_map_dict, **kwargs):
        r"""
        convert HuggingFace map dict to MindFormers map dict.

        Args:
            hf_name_map_dict: origin weight dict.
            kwargs: additional specific kwargs.

        Returns:
            ms_name_map_dict: converted weight dict.

        """
        qkv_concat = kwargs.pop("qkv_concat", False)
        ms_name_map_dict = {}
        wq_keys = []
        w1_keys = []

        for k, v in hf_name_map_dict.items():
            k = self.convert_name(k)
            ms_name_map_dict.update({k: v})
            if qkv_concat:
                part = k.split('.')
                if part[-2] == 'linear_q':
                    wq_keys.append(k)
                if part[-2] == 'gating':
                    w1_keys.append(k)

        if qkv_concat:
            for wq_key in wq_keys:
                wk_key = wq_key.replace('linear_q', 'linear_k')
                wv_key = wq_key.replace('linear_q', 'linear_v')
                wq_value = ms_name_map_dict.pop(wq_key)
                ms_name_map_dict.pop(wk_key)
                ms_name_map_dict.pop(wv_key)

                w_qkv_key = wq_key.replace('linear_q', 'linear_qkv')
                w_qkv_value = wq_value
                ms_name_map_dict.update({w_qkv_key: w_qkv_value})

            for w1_key in w1_keys:
                w3_key = w1_key.replace('gating', 'hidden')
                w1_value = ms_name_map_dict.pop(w1_key)
                ms_name_map_dict.pop(w3_key)

                w_gate_hidden_key = w1_key.replace('gating', 'linear_fc1')
                w_gate_hidden_value = w1_value
                ms_name_map_dict.update({w_gate_hidden_key: w_gate_hidden_value})

        return ms_name_map_dict
