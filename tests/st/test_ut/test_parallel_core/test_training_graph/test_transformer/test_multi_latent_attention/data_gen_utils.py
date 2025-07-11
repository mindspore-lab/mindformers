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


def get_init_params(loc, scale, config, batch_size):
    """
    Generates initial parameters for Multi-head Latent Attention (MLA).
    Input shape is (seq_length, batch_size, hidden_size).
    """
    rng = np.random.default_rng(42)
    hidden_state = rng.normal(loc=loc, scale=scale, size=(config.seq_length, batch_size, config.hidden_size))
    megatron_state_dict = {
        "linear_kv_down_proj.weight": rng.normal(
            loc=loc, scale=scale, size=(config.kv_lora_rank + config.qk_pos_emb_head_dim, config.hidden_size)),
        "linear_kv_up_proj.weight": rng.normal(
            loc=loc, scale=scale,
            size=(config.num_attention_heads * (config.qk_head_dim + config.v_head_dim), config.kv_lora_rank)
        ),
        "linear_proj.weight": rng.normal(
            loc=loc, scale=scale, size=(config.hidden_size, config.v_head_dim * config.num_attention_heads)),
        "kv_layernorm.weight": rng.normal(loc=loc, scale=scale, size=config.kv_lora_rank),
    }

    if config.q_lora_rank:
        megatron_state_dict["linear_q_down_proj.weight"] = rng.normal(
            loc=loc, scale=scale, size=(config.q_lora_rank, config.hidden_size))
        megatron_state_dict["linear_q_up_proj.weight"] = rng.normal(
            loc=loc, scale=scale,
            size=(config.num_attention_heads * (config.qk_head_dim + config.qk_pos_emb_head_dim), config.q_lora_rank),
        )
        megatron_state_dict["q_layernorm.weight"] = rng.normal(loc=loc, scale=scale, size=config.q_lora_rank)
    else:
        megatron_state_dict["linear_q_proj.weight"] = rng.normal(
            loc=loc, scale=scale,
            size=(config.num_attention_heads * (config.qk_head_dim + config.qk_pos_emb_head_dim), config.hidden_size),
        )

    mindspeed_state_dict = {
        "linear_kvb.weight": megatron_state_dict["linear_kv_up_proj.weight"],
        "linear_proj.weight": megatron_state_dict["linear_proj.weight"],
        "k_layernorm.weight": megatron_state_dict["kv_layernorm.weight"],
    }

    if config.q_lora_rank:
        mindspeed_state_dict["linear_qkv.weight"] = np.concatenate((
            megatron_state_dict["linear_q_down_proj.weight"],
            megatron_state_dict["linear_kv_down_proj.weight"]
        ))
        mindspeed_state_dict["linear_qb.weight"] = megatron_state_dict["linear_q_up_proj.weight"]
        mindspeed_state_dict["q_layernorm.weight"] = megatron_state_dict["q_layernorm.weight"]
    else:
        mindspeed_state_dict["linear_qkv.weight"] = np.concatenate((
            megatron_state_dict["linear_q_proj.weight"],
            megatron_state_dict["linear_kv_down_proj.weight"]
        ))

    attention_mask = np.ones((batch_size, 1, config.seq_length, config.seq_length), dtype=bool)
    rotary_pos_emb = np.asarray([[[[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]],
                                 [[[1.00000000e+00, 2.50000012e-04, 1.00000000e+00, 2.50000012e-04]]],
                                 [[[2.00000000e+00, 5.00000024e-04, 2.00000000e+00, 5.00000024e-04]]],
                                 [[[3.00000000e+00, 7.50000007e-04, 3.00000000e+00, 7.50000007e-04]]]])
    return hidden_state, megatron_state_dict, mindspeed_state_dict, attention_mask, rotary_pos_emb


def get_gpu_datas() -> dict[str, np.ndarray]:
    """Generate gpu data for test."""
    q8_flash_ql_kl = [[[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                        0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                        -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                        0.00063705],
                       [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                        -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                        -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]],

                      [[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                        0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                        -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                        0.00063705],
                       [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                        -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                        -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]],

                      [[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                        0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                        -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                        0.00063705],
                       [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                        -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                        -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]],

                      [[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                        0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                        -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                        0.00063705],
                       [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                        -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                        -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]]]
    q0_flash_ql_kl = [[[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                        0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                        -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                        0.00063705],
                       [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                        -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                        -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]],

                      [[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                        0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                        -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                        0.00063705],
                       [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                        -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                        -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]],

                      [[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                        0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                        -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                        0.00063705],
                       [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                        -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                        -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]],

                      [[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                        0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                        -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                        0.00063705],
                       [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                        -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                        -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]]]
    q8_flash_kl = [[[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                     0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                     -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                     0.00063705],
                    [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                     -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                     -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                     0.00756836]],

                   [[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                     0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                     -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                     0.00063705],
                    [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                     -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                     -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                     0.00756836]],

                   [[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                     0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                     -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                     0.00063705],
                    [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                     -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                     -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                     0.00756836]],

                   [[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                     0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                     -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                     0.00063705],
                    [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                     -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                     -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                     0.00756836]]]
    q8_flash_ql_kl_mscale = [[[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                               0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                               -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                               0.00063705],
                              [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                               -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                               -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                               0.00756836]],

                             [[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                               0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                               -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                               0.00063705],
                              [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                               -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                               -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                               0.00756836]],

                             [[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                               0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                               -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                               0.00063705],
                              [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                               -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                               -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                               0.00756836]],

                             [[-0.00221252, -0.00332642, -0.00065231, -0.00078201, -0.00337219,
                               0.00312805, 0.00157166, -0.00262451, -0.00154114, -0.00549316,
                               -0.00117493, 0.00473022, 0.00396729, 0.00215149, 0.00233459,
                               0.00063705],
                              [0.00430298, 0.00549316, 0.00195312, 0.00224304, 0.02026367,
                               -0.01123047, -0.00744629, 0.01275635, 0.00769043, 0.01556396,
                               -0.00270081, -0.01525879, -0.01501465, -0.01220703, 0.01214600,
                               0.00756836]]]

    return {
        "q8_flash_ql_kl": np.array(q8_flash_ql_kl),
        "q0_flash_ql_kl": np.array(q0_flash_ql_kl),
        "q8_flash_kl": np.array(q8_flash_kl),
        "q8_flash_ql_kl_mscale": np.array(q8_flash_ql_kl_mscale)
    }


def get_golden() -> dict[str, np.ndarray]:
    """Generate golden data for test."""
    q8_flash_ql_kl = [[[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                        0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                        -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                        0.00059891],
                       [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                        -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                        -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]],

                      [[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                        0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                        -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                        0.00059891],
                       [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                        -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                        -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]],

                      [[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                        0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                        -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                        0.00059891],
                       [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                        -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                        -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]],

                      [[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                        0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                        -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                        0.00059891],
                       [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                        -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                        -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]]]
    q0_flash_ql_kl = [[[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                        0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                        -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                        0.00059891],
                       [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                        -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                        -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]],

                      [[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                        0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                        -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                        0.00059891],
                       [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                        -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                        -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]],

                      [[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                        0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                        -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                        0.00059891],
                       [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                        -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                        -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]],

                      [[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                        0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                        -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                        0.00059891],
                       [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                        -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                        -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                        0.00756836]]]
    q8_flash_kl = [[[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                     0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                     -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                     0.00059891],
                    [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                     -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                     -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                     0.00756836]],

                   [[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                     0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                     -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                     0.00059891],
                    [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                     -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                     -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                     0.00756836]],

                   [[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                     0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                     -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                     0.00059891],
                    [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                     -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                     -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                     0.00756836]],

                   [[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                     0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                     -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                     0.00059891],
                    [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                     -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                     -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                     0.00756836]]]
    q8_flash_ql_kl_mscale = [[[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                               0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                               -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                               0.00059891],
                              [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                               -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                               -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                               0.00756836]],

                             [[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                               0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                               -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                               0.00059891],
                              [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                               -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                               -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                               0.00756836]],

                             [[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                               0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                               -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                               0.00059891],
                              [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                               -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                               -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                               0.00756836]],

                             [[-0.00218201, -0.00332642, -0.00063324, -0.00079727, -0.00341797,
                               0.00314331, 0.00164032, -0.00270081, -0.00156403, -0.00549316,
                               -0.00119781, 0.00479126, 0.00402832, 0.00221252, 0.00230408,
                               0.00059891],
                              [0.00433350, 0.00561523, 0.00198364, 0.00219727, 0.02026367,
                               -0.01123047, -0.00738525, 0.01281738, 0.00775146, 0.01562500,
                               -0.00265503, -0.01538086, -0.01501465, -0.01220703, 0.01214600,
                               0.00756836]]]

    return {
        "q8_flash_ql_kl": np.array(q8_flash_ql_kl),
        "q0_flash_ql_kl": np.array(q0_flash_ql_kl),
        "q8_flash_kl": np.array(q8_flash_kl),
        "q8_flash_ql_kl_mscale": np.array(q8_flash_ql_kl_mscale)
    }


GOLDEN_DATA = get_golden()
GPU_DATA = get_gpu_datas()
