#  Copyright 2025 Huawei Technologies Co., Ltd
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""Get reference data."""

import numpy as np


def get_init_params(config, loc, scale, seq_length, batch_size, vocab_size):
    """
    Generates initial parameters for Multi-Token Prediction (MTP).
    Input shape is (seq_length, batch_size, hidden_size).
    """
    rng = np.random.default_rng(42)
    data = list(range(seq_length))
    hidden_size = config.hidden_size
    mlp_size = config.ffn_hidden_size * 2 if config.gated_linear_unit else config.ffn_hidden_size

    state_dict = {
        # mtp layer weights
        "layers.0.enorm.weight": rng.normal(loc=loc, scale=scale, size=hidden_size),
        "layers.0.hnorm.weight": rng.normal(loc=loc, scale=scale, size=hidden_size),
        "layers.0.eh_proj.weight": rng.normal(loc=loc, scale=scale, size=(hidden_size, hidden_size * 2)),
        "layers.0.final_layernorm.weight": rng.normal(loc=loc, scale=scale, size=hidden_size),

        # transformer weights
        "layers.0.transformer_layer.input_layernorm.weight": rng.normal(loc=loc, scale=scale, size=hidden_size),
        "layers.0.transformer_layer.self_attention.linear_proj.weight": rng.normal(loc=loc, scale=scale,
                                                                                   size=(hidden_size, hidden_size)),
        "layers.0.transformer_layer.self_attention.linear_qkv.weight": rng.normal(loc=loc, scale=scale,
                                                                                  size=(hidden_size * 3, hidden_size)),
        "layers.0.transformer_layer.pre_mlp_layernorm.weight": rng.normal(loc=loc, scale=scale, size=hidden_size),
        "layers.0.transformer_layer.mlp.linear_fc1.weight": rng.normal(loc=loc, scale=scale,
                                                                       size=(mlp_size, hidden_size)),
        "layers.0.transformer_layer.mlp.linear_fc2.weight": rng.normal(loc=loc, scale=scale,
                                                                       size=(hidden_size, mlp_size)),
        "layers.1.transformer_layer.input_layernorm.weight": rng.normal(loc=loc, scale=scale, size=hidden_size),
        "layers.1.transformer_layer.self_attention.linear_proj.weight": rng.normal(loc=loc, scale=scale,
                                                                                   size=(hidden_size, hidden_size)),
        "layers.1.transformer_layer.self_attention.linear_qkv.weight": rng.normal(loc=loc, scale=scale,
                                                                                  size=(hidden_size * 3, hidden_size)),
        "layers.1.transformer_layer.pre_mlp_layernorm.weight": rng.normal(loc=loc, scale=scale, size=hidden_size),
        "layers.1.transformer_layer.mlp.linear_fc1.weight": rng.normal(loc=loc, scale=scale,
                                                                       size=(mlp_size, hidden_size)),
        "layers.1.transformer_layer.mlp.linear_fc2.weight": rng.normal(loc=loc, scale=scale,
                                                                       size=(hidden_size, mlp_size)),
        # input layers weights
        "word_embeddings.weight": rng.normal(loc=loc, scale=scale, size=(vocab_size, hidden_size)),
        "position_embeddings.weight": rng.normal(loc=loc, scale=scale, size=(seq_length, hidden_size)),
        "weight": rng.normal(loc=loc, scale=scale, size=(vocab_size, hidden_size)),  # output_layer
    }
    hidden_states = rng.normal(loc=loc, scale=scale, size=(seq_length, batch_size, hidden_size))

    # Will be transposed to [s, b, h] in LanguageModelEmbedding
    input_ids = np.tile(data, (batch_size, 1))
    position_ids = np.tile(data, (batch_size, 1))

    labels = (1 + np.tile(data, (batch_size, 1)))
    loss_mask = np.ones((batch_size, seq_length))

    attention_mask = np.ones((batch_size, 1, seq_length, seq_length), dtype=bool)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
        'hidden_states': hidden_states,
        'state_dict': state_dict
    }


def get_gpu_datas() -> dict[str, np.ndarray]:
    """Generate gpu data for test."""
    single_card_baseline = [2.2938010692596436]
    pe_rope = [2.3010997772216797]
    return {
        "single_card_baseline": np.array(single_card_baseline),
        "pe_rope": np.array(pe_rope),
    }


def get_golden() -> dict[str, np.ndarray]:
    """Generate golden data for test."""
    single_card_baseline = [2.293926239013672]
    pe_rope = [2.3010616302490234]
    return {
        "single_card_baseline": np.array(single_card_baseline),
        "pe_rope": np.array(pe_rope),
    }


GOLDEN_DATA = get_golden()
GPU_DATA = get_gpu_datas()
