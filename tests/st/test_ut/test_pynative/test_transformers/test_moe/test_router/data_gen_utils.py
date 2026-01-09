# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Generate data for TokenChoiceTopKRouter test."""

import numpy as np


def get_init_params(hidden_size=32, num_experts=4, batch_size=2, seq_length=4):
    """Generate initialization parameters."""
    # Generate initialization parameters
    np.random.seed(42)
    # inputs: (batch_size, seq_length, hidden_size) -> (bs, seq_len, dim)
    inputs = 0.01 * np.random.randn(batch_size, seq_length, hidden_size).astype(np.float32)
    # gate.weight: (num_experts, hidden_size)
    gate_weight = 0.01 * np.random.randn(num_experts, hidden_size).astype(np.float32)
    # expert_bias: (num_experts,)
    expert_bias = 0.01 * np.random.randn(num_experts).astype(np.float32)

    return {"inputs": inputs, "weight": gate_weight, "export_bias": expert_bias}


def get_golden() -> dict[str, np.ndarray]:
    """Generate data for get_golden (Auto-generated)."""
    return {
        "num_tokens_per_expert_case0_1_2_3_4_5_6_7": np.array([0.0, 0.0, 8.0, 8.0], dtype=np.float32),
        "num_tokens_per_expert_case8": np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32),
        "selected_experts_indices_case0_1_2_3_4_5_6_7": np.array(
            [[3.0, 2.0], [3.0, 2.0], [3.0, 2.0], [3.0, 2.0], [3.0, 2.0], [3.0, 2.0], [3.0, 2.0], [3.0, 2.0]],
            dtype=np.float32,
        ),
        "selected_experts_indices_case8": np.array(
            [[0.0, 1.0], [2.0, 3.0], [0.0, 1.0], [2.0, 3.0], [0.0, 1.0], [2.0, 3.0], [0.0, 1.0], [2.0, 3.0]],
            dtype=np.float32,
        ),
        "top_scores_case0_6": np.array(
            [
                [0.25001216, 0.250293],
                [0.2501407, 0.24987544],
                [0.25010905, 0.25015932],
                [0.25009653, 0.24982247],
                [0.24991079, 0.24995713],
                [0.25000516, 0.24999422],
                [0.24985604, 0.24979976],
                [0.24989985, 0.24988692],
            ],
            dtype=np.float32,
        ),
        "top_scores_case1_7": np.array(
            [
                [0.49999082, 0.5002715],
                [0.5001504, 0.4998851],
                [0.5000666, 0.5001168],
                [0.50014025, 0.49986616],
                [0.4999318, 0.49997818],
                [0.49993873, 0.49992782],
                [0.49992365, 0.49986735],
                [0.49986008, 0.4998471],
            ],
            dtype=np.float32,
        ),
        "top_scores_case2": np.array(
            [
                [0.49971932, 0.5002806],
                [0.50026524, 0.49973473],
                [0.49994978, 0.5000503],
                [0.5002741, 0.4997259],
                [0.49995366, 0.5000464],
                [0.50001097, 0.49998906],
                [0.5000563, 0.4999437],
                [0.50001293, 0.49998707],
            ],
            dtype=np.float32,
        ),
        "top_scores_case3": np.array(
            [
                [0.49985972, 0.5001403],
                [0.5001326, 0.49986735],
                [0.4999749, 0.50002515],
                [0.50013703, 0.49986294],
                [0.4999768, 0.5000232],
                [0.5000055, 0.49999455],
                [0.50002813, 0.49997184],
                [0.5000065, 0.4999935],
            ],
            dtype=np.float32,
        ),
        "top_scores_case4": np.array(
            [
                [0.5000243, 0.500586],
                [0.5002814, 0.49975088],
                [0.5002181, 0.50031865],
                [0.50019306, 0.49964494],
                [0.49982157, 0.49991426],
                [0.5000103, 0.49998844],
                [0.49971208, 0.49959952],
                [0.4997997, 0.49977383],
            ],
            dtype=np.float32,
        ),
        "top_scores_case5": np.array(
            [
                [0.24999541, 0.25013575],
                [0.2500752, 0.24994256],
                [0.2500333, 0.2500584],
                [0.25007012, 0.24993308],
                [0.2499659, 0.24998909],
                [0.24996936, 0.24996391],
                [0.24996182, 0.24993367],
                [0.24993004, 0.24992356],
            ],
            dtype=np.float32,
        ),
        "top_scores_case8": np.array(
            [
                [0.24982317, 0.24987176],
                [0.24987544, 0.2501407],
                [0.24976742, 0.24996418],
                [0.24982247, 0.25009653],
                [0.25020763, 0.24992438],
                [0.24999422, 0.25000516],
                [0.2501769, 0.2501673],
                [0.24988692, 0.24989985],
            ],
            dtype=np.float32,
        ),
    }


def get_gpu_datas() -> dict[str, np.ndarray]:
    """Generate data for get_gpu_datas (Auto-generated)."""
    return {
        "num_tokens_per_expert_case0_1_2_3_4_5_6_7": np.array([0.0, 0.0, 8.0, 8.0], dtype=np.float32),
        "num_tokens_per_expert_case8": np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32),
        "selected_experts_indices_case0_1_2_3_4_5_6_7": np.array(
            [[3.0, 2.0], [3.0, 2.0], [3.0, 2.0], [3.0, 2.0], [3.0, 2.0], [3.0, 2.0], [3.0, 2.0], [3.0, 2.0]],
            dtype=np.float32,
        ),
        "selected_experts_indices_case8": np.array(
            [[0.0, 1.0], [2.0, 3.0], [0.0, 1.0], [2.0, 3.0], [0.0, 1.0], [2.0, 3.0], [0.0, 1.0], [2.0, 3.0]],
            dtype=np.float32,
        ),
        "top_scores_case0_6": np.array(
            [
                [0.25001222, 0.25029227],
                [0.2501401, 0.24987544],
                [0.25010988, 0.25015855],
                [0.25009608, 0.24982241],
                [0.24991027, 0.2499571],
                [0.25000548, 0.24999452],
                [0.24985537, 0.24979962],
                [0.24989949, 0.24988711],
            ],
            dtype=np.float32,
        ),
        "top_scores_case1_7": np.array(
            [
                [0.49999094, 0.5002708],
                [0.5001497, 0.4998851],
                [0.50006676, 0.50011533],
                [0.5001402, 0.49986652],
                [0.49993134, 0.49997818],
                [0.49993896, 0.499928],
                [0.49992323, 0.4998674],
                [0.49985984, 0.4998474],
            ],
            dtype=np.float32,
        ),
        "top_scores_case2": np.array(
            [
                [0.49972016, 0.5002799],
                [0.50026464, 0.49973533],
                [0.49995133, 0.50004864],
                [0.5002737, 0.4997263],
                [0.49995315, 0.50004685],
                [0.50001097, 0.49998903],
                [0.5000558, 0.4999442],
                [0.5000124, 0.4999876],
            ],
            dtype=np.float32,
        ),
        "top_scores_case3": np.array(
            [
                [0.49986008, 0.50013983],
                [0.5001323, 0.4998677],
                [0.49997568, 0.50002426],
                [0.50013685, 0.49986318],
                [0.49997658, 0.5000234],
                [0.5000055, 0.49999452],
                [0.5000279, 0.49997208],
                [0.50000626, 0.4999938],
            ],
            dtype=np.float32,
        ),
        "top_scores_case4": np.array(
            [
                [0.50002444, 0.50058454],
                [0.5002802, 0.49975088],
                [0.50021976, 0.5003171],
                [0.50019217, 0.49964482],
                [0.49982053, 0.4999142],
                [0.50001097, 0.49998903],
                [0.49971074, 0.49959925],
                [0.49979898, 0.49977422],
            ],
            dtype=np.float32,
        ),
        "top_scores_case5": np.array(
            [
                [0.24999547, 0.2501354],
                [0.25007486, 0.24994256],
                [0.25003338, 0.25005767],
                [0.2500701, 0.24993326],
                [0.24996567, 0.24998909],
                [0.24996948, 0.249964],
                [0.24996161, 0.2499337],
                [0.24992992, 0.2499237],
            ],
            dtype=np.float32,
        ),
        "top_scores_case8": np.array(
            [
                [0.24982391, 0.24987157],
                [0.24987544, 0.2501401],
                [0.24976665, 0.24996492],
                [0.24982241, 0.25009608],
                [0.25020882, 0.24992386],
                [0.24999452, 0.25000548],
                [0.25017822, 0.25016677],
                [0.24988711, 0.24989949],
            ],
            dtype=np.float32,
        ),
    }


GOLDEN_DATA = get_golden()
GPU_DATA = get_gpu_datas()
