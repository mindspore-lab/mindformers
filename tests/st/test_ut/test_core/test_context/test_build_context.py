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
"""Test build_context.py"""
import multiprocessing
import os

import pytest
from mindformers.core.context import (
    build_context,
    build_mf_context,
    get_context,
    set_context,
)

def get_config_tpl():
    return {'context': {'mode': 'PYNATIVE_MODE'}, 'parallel': {}}

def run_in_subprocess(func, *args):
    """Run testcase in subprocess and check it is successfully."""
    process = multiprocessing.Process(target=func, args=args)
    process.start()
    process.join()
    assert process.exitcode == 0


def run_deterministic_setting(
        mode, switch, hccl_deterministic_env, te_parallel_compiler_env,
        custom_matmul_shuffle_env, lccl_deterministic_env,
        hccl_deterministic_expect, te_parallel_compiler_expect,
        custom_matmul_shuffle_expect, lccl_deterministic_expect):
    """Execute deterministic setting testcase."""
    os.environ.clear()
    env = {
        'HCCL_DETERMINISTIC': hccl_deterministic_env,
        'TE_PARALLEL_COMPILER': te_parallel_compiler_env,
        'CUSTOM_MATMUL_SHUFFLE': custom_matmul_shuffle_env,
        'LCCL_DETERMINISTIC': lccl_deterministic_env,
    }
    os.environ.update({k: v for k, v in env.items() if v is not None})
    config_tpl = get_config_tpl()
    config_tpl['run_mode'] = mode
    build_context(config_tpl)
    if mode in ('train', 'finetune'):
        set_context(train_precision_sync=switch)
        assert get_context('train_precision_sync') == switch
    else:
        set_context(infer_precision_sync=switch)
        assert get_context('infer_precision_sync') == switch
    assert os.getenv('HCCL_DETERMINISTIC') == hccl_deterministic_expect
    assert os.getenv('TE_PARALLEL_COMPILER') == te_parallel_compiler_expect
    assert os.getenv('CUSTOM_MATMUL_SHUFFLE') == custom_matmul_shuffle_expect
    assert os.getenv('LCCL_DETERMINISTIC') == lccl_deterministic_expect

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize(
    'mode, switch, hccl_deterministic_env, te_parallel_compiler_env, '
    'custom_matmul_shuffle_env, lccl_deterministic_env, '
    'hccl_deterministic_expect, te_parallel_compiler_expect, '
    'custom_matmul_shuffle_expect, lccl_deterministic_expect', (
        ('train', False, 'true', '1', None, None, None, None, 'on', '0'),
        ('train', False, 'false', '0', None, None, 'false', '0', 'on', '0'),
        ('train', True, 'false', '0', None, None, 'true', '1', 'on', '0'),
        ('finetune', True, 'false', '0', None, None, 'true', '1', 'on', '0'),
        ('predict', False, None, None, 'off', '1', None, None, 'on', '0'),
        ('predict', True, None, None, 'on', '0', 'true', '1', 'off', '1'),
    )
)
def test_deterministic(mode, switch, hccl_deterministic_env,
                       te_parallel_compiler_env, custom_matmul_shuffle_env,
                       lccl_deterministic_env, hccl_deterministic_expect,
                       te_parallel_compiler_expect,
                       custom_matmul_shuffle_expect,
                       lccl_deterministic_expect):
    """
    Feature: Test deterministic computing setting through set_context().
    Description: Compare the setting env variables and expected variables.
    Expectation: setting env variables and expected variables is different.
    """
    run_in_subprocess(run_deterministic_setting, mode, switch,
                      hccl_deterministic_env, te_parallel_compiler_env,
                      custom_matmul_shuffle_env, lccl_deterministic_env,
                      hccl_deterministic_expect, te_parallel_compiler_expect,
                      custom_matmul_shuffle_expect, lccl_deterministic_expect)

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_build_mf_context():
    """
    Feature: Test whether MFContextOperator is a singleton.
    Description: The MFContextOperator instance created twice is the same object.
    Expectation: The MFContextOperator instance created twice are different object.
    """
    config_tpl = get_config_tpl()
    mf_ctx = build_mf_context(config_tpl)
    another_mf_ctx = build_mf_context(config_tpl)
    assert mf_ctx is another_mf_ctx
