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
"""Test save/load common info."""

import os
import shutil
import pytest

import mindspore as ms

from mindformers.checkpoint.sharded_tensor import get_sharded_tensor_list_from_strategy_metadata
from mindformers.checkpoint.metadata import save_metadata, load_metadata
from mindformers.checkpoint.utils import (
    get_checkpoint_iter_dir,
    get_metadata_filename,
    get_checkpoint_name
)

AA = ms.parallel.Layout((2, 2, 2), ("dp", "sp", "mp"))
A = AA("dp", "mp")

GLOBAL_STRATEGY_INFO = {
    0: {
        "decoder.layers.0.input_layernorm.weight": [A, 'float32', [3584]],
        "adam_m.decoder.layers.0.input_layernorm.weight": [A, 'float32', [3584]],
    },
    1: {
        "decoder.layers.0.input_layernorm.weight": [A, 'float32', [3584]],
        "adam_m.decoder.layers.0.input_layernorm.weight": [A, 'float32', [3584]],
    }
}
MODEL_KEYS = ["decoder.layers.0.input_layernorm.weight"]
USER_PREFIX = "my_test_net"
CHECKPOINT_ROOT_DIR = "./output_megatron_format_metadata"

ITERATION_WITH_OPTIMIZER = 1
ITERATION_WITHOUT_OPTIMIZER = 2
NOT_EXISTS = False


def save_metadata_without_npu(global_strategy_info, model_keys, user_prefix, metadata_file_path, save_optimizer):
    """Saving metadata.json without NPU ranks, using mock data."""
    npu_nums = 2
    sharded_tensor_metas = list()
    param_file_mappings = list()

    for cur_npu_rank in range(0, npu_nums):
        org_cur_rank_strategy_layout = global_strategy_info[cur_npu_rank]
        cur_rank_strategy_layout = [
            dict([item])
            for item in org_cur_rank_strategy_layout.items()
        ]

        # Get Sharded tensors from strategy metadata of current rank.
        cur_rank_sharded_tensors = get_sharded_tensor_list_from_strategy_metadata(
            param_infos=cur_rank_strategy_layout,
            cur_npu_rank=cur_npu_rank,
            filter_func=(lambda x: x in list(model_keys)) if not save_optimizer else None
        )

        # Get mappings of parameter file of current rank.
        for sharded_tensor in cur_rank_sharded_tensors:
            if save_optimizer and sharded_tensor.key not in list(model_keys):
                ckpt_name = get_checkpoint_name(None, user_prefix, cur_npu_rank, npu_nums, 'Optimizer')
            else:
                ckpt_name = get_checkpoint_name(None, user_prefix, cur_npu_rank, npu_nums, 'Model')
            param_file_mappings.append(
                (
                    ckpt_name + '.safetensors',
                    cur_npu_rank,
                    (sharded_tensor.key, sharded_tensor.global_offset)
                )
            )

        sharded_tensor_metas.append(cur_rank_sharded_tensors)

    save_metadata(sharded_tensor_metas, param_file_mappings, metadata_file_path)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_save_and_load_metadata_case():
    """
    Feature: Test save metadata info in none-has optimizer two cases, then load them.
    Description: Simulate saving 'metadata.json' in succession
        to ensure that the paths and contents of both accesses are normal.
        Then load the saved metadata to check whether the load function can obtain the value normally.
    Expectation: No error is reported during test this case.
    """
    # 1. Test save the 'metadata.json' with optimizer info.
    has_optimizer_checkpoint_path = get_checkpoint_iter_dir(CHECKPOINT_ROOT_DIR, ITERATION_WITH_OPTIMIZER)
    os.makedirs(has_optimizer_checkpoint_path, exist_ok=True)

    has_optimizer_metadata_file_path = get_metadata_filename(CHECKPOINT_ROOT_DIR, ITERATION_WITH_OPTIMIZER)
    save_metadata_without_npu(
        global_strategy_info=GLOBAL_STRATEGY_INFO,
        model_keys=MODEL_KEYS,
        user_prefix=USER_PREFIX,
        metadata_file_path=has_optimizer_metadata_file_path,
        save_optimizer=True
    )

    assert os.path.isfile(has_optimizer_metadata_file_path)

    # 2. Test save the 'metadata.json' without optimizer info.
    no_optimizer_checkpoint_path = get_checkpoint_iter_dir(CHECKPOINT_ROOT_DIR, ITERATION_WITHOUT_OPTIMIZER)
    os.makedirs(no_optimizer_checkpoint_path, exist_ok=True)

    no_optimizer_metadata_file_path = get_metadata_filename(CHECKPOINT_ROOT_DIR, ITERATION_WITHOUT_OPTIMIZER)
    save_metadata_without_npu(
        global_strategy_info=GLOBAL_STRATEGY_INFO,
        model_keys=MODEL_KEYS,
        user_prefix=USER_PREFIX,
        metadata_file_path=no_optimizer_metadata_file_path,
        save_optimizer=False
    )

    assert os.path.isfile(no_optimizer_metadata_file_path)

    # 3. Test load 'metadata.json' with optimizer info.
    has_optimizer_sharded_tensors, has_optimizer_param_file_mappings = load_metadata(
        checkpoints_path=CHECKPOINT_ROOT_DIR,
        iteration=ITERATION_WITH_OPTIMIZER
    )

    decoder_input_0 = has_optimizer_sharded_tensors[('decoder.layers.0.input_layernorm.weight', (0,))]
    assert decoder_input_0.local_shape == (1792,)
    assert decoder_input_0.global_shape == (3584,)
    assert decoder_input_0.global_offset == (0,)

    decoder_input_1 = has_optimizer_sharded_tensors[('decoder.layers.0.input_layernorm.weight', (1,))]
    assert decoder_input_1.local_shape == (1792,)
    assert decoder_input_1.global_shape == (3584,)
    assert decoder_input_1.global_offset == (1,)

    adam_input_layernorm_0 = has_optimizer_sharded_tensors[('adam_m.decoder.layers.0.input_layernorm.weight', (0,))]
    assert adam_input_layernorm_0 is not None
    adam_input_layernorm_1 = has_optimizer_sharded_tensors[('adam_m.decoder.layers.0.input_layernorm.weight', (1,))]
    assert adam_input_layernorm_1 is not None

    adam_mapping_0 = has_optimizer_param_file_mappings["('adam_m.decoder.layers.0.input_layernorm.weight', (0,))"][0]
    assert adam_mapping_0["storage_rank"] == 0
    assert adam_mapping_0["file_name"] == "my_test_net-opt-0000000-0000002.safetensors"

    # 4. Test load 'metadata.json' without optimizer info.
    no_optimizer_sharded_tensors, no_optimizer_param_file_mappings = load_metadata(
        checkpoints_path=CHECKPOINT_ROOT_DIR,
        iteration=ITERATION_WITHOUT_OPTIMIZER
    )

    for k in no_optimizer_sharded_tensors.keys():
        assert "adam" not in k[0]

    decoder_input_1_no_op = no_optimizer_sharded_tensors[('decoder.layers.0.input_layernorm.weight', (1,))]
    assert decoder_input_1_no_op.local_shape == (1792,)
    assert decoder_input_1_no_op.global_shape == (3584,)
    assert decoder_input_1_no_op.global_offset == (1,)

    decoder_mapping_1 = no_optimizer_param_file_mappings["('decoder.layers.0.input_layernorm.weight', (1,))"][0]
    assert decoder_mapping_1["storage_rank"] == 1
    assert decoder_mapping_1["file_name"] == "my_test_net-model-0000001-0000002.safetensors"

    # Clear all save files for test
    shutil.rmtree(CHECKPOINT_ROOT_DIR)
    assert os.path.exists(CHECKPOINT_ROOT_DIR) == NOT_EXISTS
    assert os.path.exists(has_optimizer_metadata_file_path) == NOT_EXISTS
    assert os.path.exists(no_optimizer_metadata_file_path) == NOT_EXISTS
