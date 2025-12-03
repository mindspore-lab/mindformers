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
"""Test parallel decoding"""
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor

from mindformers.generation.parallel_decoding import (
    _logits_process,
    _pre_process,
    _la_logits_process,
    _la_pre_process,
    _memory_decoding_pre_process,
    _prefix_cache_pre_process,
    parallel_decoding_control,
    parallel_decoding_logits_process,
    _construct_mask,
    _parallel_decoding_pad,
    _parallel_decoding_pad_2d_tensor
)


class MockConfig:
    def __init__(self, parallel_decoding=None):
        if parallel_decoding:
            self.parallel_decoding_params = {"parallel_decoding": parallel_decoding}
        else:
            self.parallel_decoding_params = None


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_register_decorators():
    """
    Feature: parallel decoding.
    Description: test a function in parallel decoding.
    Expectation: success.
    """
    assert 'la' in _logits_process
    assert 'la' in _pre_process
    assert 'memory_decoding' in _pre_process
    assert 'prefix_cache' in _pre_process


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_construct_mask():
    """
    Feature: parallel decoding.
    Description: test a function in parallel decoding.
    Expectation: success.
    """
    q_seq_lens = [2, 3]
    mask = _construct_mask(q_seq_lens)
    expected = np.array([
        [-0, 1, 1, 1, 1],
        [-0, -0, 1, 1, 1],
        [1, 1, -0, 1, 1],
        [1, 1, -0, -0, 1],
        [1, 1, -0, -0, -0]
    ], dtype=np.float16)
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parallel_decoding_pad():
    """
    Feature: parallel decoding.
    Description: test a function in parallel decoding.
    Expectation: success.
    """
    arr = np.array([1, 2, 3])
    padded = _parallel_decoding_pad(arr, axis=0, pad_len=5, value=-1)
    expected = np.array([1, 2, 3, -1, -1])
    np.testing.assert_array_equal(padded, expected)

    # pad_len < current len → no change
    same = _parallel_decoding_pad(arr, axis=0, pad_len=2, value=-1)
    np.testing.assert_array_equal(same, arr)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parallel_decoding_pad_2d_tensor():
    """
    Feature: parallel decoding.
    Description: test a function in parallel decoding.
    Expectation: success.
    """
    inputs = np.array([1, 2, 3, 4, 5, 6])
    lens = [2, 3]
    padded = _parallel_decoding_pad_2d_tensor(inputs, pad_seq_len=4, lens=lens, value=-1)
    expected = np.array([
        [1, 2, -1, -1],
        [3, 4, 5, -1]
    ])
    np.testing.assert_array_equal(padded, expected)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_la_logits_process_simple():
    """
    Feature: parallel decoding.
    Description: test a function in parallel decoding.
    Expectation: success.
    """
    logits = Tensor(np.random.rand(4, 100), ms.float32)
    result = _la_logits_process(logits, None, None, False)
    assert result.shape == (4, 100)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_la_logits_process_with_q_seq_lens():
    """
    Feature: parallel decoding.
    Description: test a function in parallel decoding.
    Expectation: success.
    """
    logits = Tensor(np.random.rand(6, 100), ms.float32)  # batch=2, max_seq=3
    q_seq_lens = [2, 3]
    block_tables = [[1, 2], [3, 4]]
    result = _la_logits_process(logits, q_seq_lens, block_tables, prefill=True)
    assert result.shape == (2, 100)  # last token of each seq


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_la_pre_process_normal():
    """
    Feature: parallel decoding.
    Description: test a function in parallel decoding.
    Expectation: success.
    """
    config = MockConfig("la")
    input_ids = Tensor([[1, 2, 3]], ms.int32)
    model_inputs = {}
    block_tables = np.array([[10, 11]])
    slot_mapping = np.array([0, 1, 2])
    q_seq_lens = [3]

    out_model_inputs, out_block, out_slot = _la_pre_process(
        config, input_ids, model_inputs,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        q_seq_lens=q_seq_lens
    )

    assert isinstance(out_model_inputs['input_ids'], Tensor)
    assert out_model_inputs['input_ids'].shape == (1, 3)
    assert out_model_inputs['q_seq_lens'].shape == (1,)
    assert np.array_equal(out_block, block_tables.astype(np.int32))
    assert np.array_equal(out_slot, slot_mapping.astype(np.int32))


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_la_pre_process_with_max_padding():
    """
    Feature: parallel decoding.
    Description: test a function in parallel decoding.
    Expectation: success.
    """
    config = MockConfig("la")
    input_ids = Tensor([[1, 2, 0, 0, 3, 4, 0, 0]], ms.int32)  # shape (1,8), max_len=4, two seqs
    model_inputs = {}
    block_tables = np.array([[1, 2], [3, 4]])
    slot_mapping = np.array([0, 1, 0, 0, 2, 3, 0, 0])
    q_seq_lens = [2, 2]  # each seq has 2 real tokens

    out_model_inputs, _, _ = _la_pre_process(
        config, input_ids, model_inputs,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        q_seq_lens=q_seq_lens
    )

    # Should extract [1,2,3,4] → shape (1,4)
    assert out_model_inputs['input_ids'].shape == (1, 4)
    expected_ids = np.array([[1, 2, 0, 0]])
    np.testing.assert_array_equal(out_model_inputs['input_ids'].asnumpy(), expected_ids)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_la_pre_process_no_q_seq_lens():
    """
    Feature: parallel decoding.
    Description: test a function in parallel decoding.
    Expectation: success.
    """
    config = MockConfig("la")
    input_ids = Tensor([[1, 2, 3]], ms.int32)
    model_inputs = {}
    block_tables = np.array([[10, 11]])
    slot_mapping = np.array([0, 1, 2])

    out_model_inputs, _, _ = _la_pre_process(
        config, input_ids, model_inputs,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        q_seq_lens=None,
        valid_length_each_example=[3]
    )

    assert out_model_inputs['q_seq_lens'].shape == (1,)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_memory_and_prefix_preprocess():
    """
    Feature: parallel decoding.
    Description: test a function in parallel decoding.
    Expectation: success.
    """
    config = MockConfig("memory_decoding")
    input_ids = Tensor([], ms.int32)
    model_inputs = {}
    block_tables = np.array([0])
    slot_mapping = np.array([0])

    out1 = _memory_decoding_pre_process(config, input_ids, model_inputs,
                                        block_tables=block_tables, slot_mapping=slot_mapping)
    out2 = _prefix_cache_pre_process(config, input_ids, model_inputs,
                                     block_tables=block_tables, slot_mapping=slot_mapping)

    assert np.array_equal(out1[1], block_tables.astype(np.int32))
    assert np.array_equal(out2[2], slot_mapping.astype(np.int32))


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parallel_decoding_control():
    """
    Feature: parallel decoding.
    Description: test a function in parallel decoding.
    Expectation: success.
    """
    assert parallel_decoding_control(MockConfig("la")) is True
    assert parallel_decoding_control(MockConfig("memory_decoding")) is True
    assert parallel_decoding_control(MockConfig("prefix_cache")) is True
    assert parallel_decoding_control(MockConfig("invalid")) is False
    assert parallel_decoding_control(MockConfig(None)) is False


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parallel_decoding_logits_process():
    """
    Feature: parallel decoding.
    Description: test a function in parallel decoding.
    Expectation: success.
    """
    config = MockConfig("la")
    logits = Tensor(np.random.rand(2, 100), ms.float32)
    result = parallel_decoding_logits_process(config, logits, None, None, False)
    assert result.shape == (2, 100)
