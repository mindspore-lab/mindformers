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
"""Test the validators of context."""
from unittest.mock import patch

import pytest

from mindformers.core.context.validators import (
    validate_invalid_predict_mode,
    validate_ms_ctx_mode,
    validate_mf_ctx_run_mode,
    validate_parallel_mode,
    validate_precision_sync,
    validate_sink_size,
)
from mindformers.tools import MindFormerConfig


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize(
    'use_past, use_flash_attention, expect_error', (
        (None, True, False),
        (False, "str", False),
        (False, True, True),
    )
)
def test_validate_invalid_predict(use_past, use_flash_attention, expect_error):
    """
    Feature: Test config of use_past and use_flash_attention in predict process.
    Description: The the config whether is valid.
    Expectation: The result of execution does not equal the expected result.
    """
    config_ = {
        'run_mode': 'predict',
        'model': {
            'model_config': {
                'use_past': use_past,
                'use_flash_attention': use_flash_attention
            }
        },
    }
    cfg = MindFormerConfig(**config_)
    if expect_error:
        with pytest.raises(ValueError) as exc_info:
            validate_invalid_predict_mode(cfg)
        assert str(exc_info.value) == (
            "Conflict detected in predict mode: "
            "Flash Attention is incompatible when use_past=False")
    else:
        assert validate_invalid_predict_mode(cfg) is None


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize(
    'run_mode, expect_error', (
        (None, False),
        ('train', False),
        ('predict', False),
        ('finetune', False),
        ('eval', False),
        ('predict_with_train_model', False),
        ('invalid_mode', True),
    )
)
def test_validate_mf_ctx_run_mode(run_mode, expect_error):
    """
    Feature: Test validate_mf_ctx_run_mode function.
    Description: Test run_mode validation with different values.
    Expectation: Valid run_mode passes, invalid run_mode raises ValueError.
    """
    config_ = {'run_mode': run_mode}
    cfg = MindFormerConfig(**config_)
    if expect_error:
        with pytest.raises(ValueError) as exc_info:
            validate_mf_ctx_run_mode(cfg)
        assert 'Invalid run_mode' in str(exc_info.value)
    else:
        assert validate_mf_ctx_run_mode(cfg) is None


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize(
    'mode, expect_error', (
        ('GRAPH_MODE', False),
        ('PYNATIVE_MODE', False),
        (0, False),
        (1, False),
        ('INVALID_MODE', True),
        (999, True),
    )
)
def test_validate_ms_ctx_mode(mode, expect_error):
    """
    Feature: Test validate_ms_ctx_mode function.
    Description: Test context.mode validation with different values.
    Expectation: Valid mode passes, invalid mode raises ValueError.
    """
    config_ = {'context': {'mode': mode}}
    cfg = MindFormerConfig(**config_)
    if expect_error:
        with pytest.raises(ValueError) as exc_info:
            validate_ms_ctx_mode(cfg)
        assert 'Invalid mode' in str(exc_info.value)
    else:
        assert validate_ms_ctx_mode(cfg) is None
        # Verify mode is set
        assert cfg.get_value('context.mode') is not None


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize(
    'parallel_mode, expect_error', (
        ('DATA_PARALLEL', False),
        ('SEMI_AUTO_PARALLEL', False),
        ('AUTO_PARALLEL', False),
        ('HYBRID_PARALLEL', False),
        ('STAND_ALONE', False),
        (0, False),
        (1, False),
        (2, False),
        (3, False),
        ('INVALID_MODE', True),
        (999, True),
    )
)
def test_validate_parallel_mode(parallel_mode, expect_error):
    """
    Feature: Test validate_parallel_mode function.
    Description: Test parallel.parallel_mode validation with different values.
    Expectation: Valid parallel_mode passes, invalid parallel_mode raises ValueError.
    """
    config_ = {'parallel': {'parallel_mode': parallel_mode}}
    cfg = MindFormerConfig(**config_)
    if expect_error:
        with pytest.raises(ValueError) as exc_info:
            validate_parallel_mode(cfg)
        assert 'Invalid parallel mode' in str(exc_info.value)
    else:
        assert validate_parallel_mode(cfg) is None
        # Verify parallel_mode is set
        assert cfg.get_value('parallel.parallel_mode') is not None


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@patch('mindformers.core.context.validators.check_tft_valid')
@pytest.mark.parametrize(
    'tft_valid, sink_size, expect_error', (
        (False, 1, False),
        (False, 2, False),
        (True, 1, False),
        (True, 2, True),
    )
)
def test_validate_sink_size(mock_check_tft, tft_valid, sink_size, expect_error):
    """
    Feature: Test validate_sink_size function.
    Description: Test sink_size validation when TFT is valid.
    Expectation: sink_size must be 1 when TFT is valid.
    """
    mock_check_tft.return_value = tft_valid
    config_ = {'runner_config': {'sink_size': sink_size}}
    cfg = MindFormerConfig(**config_)
    if expect_error:
        with pytest.raises(ValueError) as exc_info:
            validate_sink_size(cfg)
        assert 'sink_size should be 1' in str(exc_info.value)
    else:
        assert validate_sink_size(cfg) is None


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize(
    'train_precision_sync, infer_precision_sync, expect_error', (
        (None, None, False),
        (True, None, False),
        (False, None, False),
        (None, True, False),
        (None, False, False),
        ('invalid', None, True),
        (None, 'invalid', True),
        (1, None, True),
        (None, 0, True),
    )
)
def test_validate_precision_sync(
    train_precision_sync, infer_precision_sync, expect_error
    ):
    """
    Feature: Test validate_precision_sync function.
    Description: Test train_precision_sync and infer_precision_sync validation.
    Expectation: Bool values pass, non-bool values raise ValueError.
    """
    config_ = {}
    if train_precision_sync is not None:
        config_['train_precision_sync'] = train_precision_sync
    if infer_precision_sync is not None:
        config_['infer_precision_sync'] = infer_precision_sync
    cfg = MindFormerConfig(**config_)
    if expect_error:
        with pytest.raises(ValueError) as exc_info:
            validate_precision_sync(cfg)
        assert 'should be bool' in str(exc_info.value)
    else:
        assert validate_precision_sync(cfg) is None
