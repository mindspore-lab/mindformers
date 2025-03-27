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
import pytest

from mindformers.core.context.validators import validate_invalid_predict_mode
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
