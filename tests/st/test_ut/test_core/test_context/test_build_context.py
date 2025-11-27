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
from unittest.mock import patch

import pytest

from mindformers.core.context import (
    build_context,
    build_mf_context,
    build_parallel_context,
    get_context,
    is_legacy_model,
    set_context,
)
from mindformers.core.context.build_context import (
    Context,
    MFContextOperator,
    MSContextOperator,
    set_cpu_affinity,
    set_ms_affinity,
)
from mindformers.tools.register import MindFormerConfig


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
def test_mf_context_singleton():
    """
    Feature: Test whether MFContextOperator is a singleton.
    Description: The MFContextOperator instance created twice is the same object.
    Expectation: The MFContextOperator instance created twice are different object.
    """
    config_tpl = get_config_tpl()
    mf_ctx = build_mf_context(config_tpl)
    another_mf_ctx = build_mf_context(config_tpl)
    assert mf_ctx is another_mf_ctx


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_context_singleton():
    """
    Feature: Test Context singleton pattern.
    Description: Test that Context is a singleton.
    Expectation: Multiple Context instances are the same object.
    """
    def is_singleton_context():
        config_tpl = get_config_tpl()
        ctx1 = build_context(config_tpl)
        ctx2 = build_context(config_tpl)
        assert ctx1 is ctx2
    run_in_subprocess(is_singleton_context)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize(
    'cfg, is_legacy_model_except', (
        ({}, True),
        ({'use_legacy': True}, True),
        ({'use_legacy': False}, False),
    )
)
def test_get_use_legacy(cfg, is_legacy_model_except):
    """
    Feature: Test whether the method of getting use_legacy is correct.
    Description: Test get_context and is_legacy_model functions.
    Expectation: The result of execution does not equal the expected result.
    """
    build_mf_context(cfg)
    assert is_legacy_model() == is_legacy_model_except
    MFContextOperator.reset_instance()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_context_set_mf_ctx_run_mode():
    """
    Feature: Test Context.set_mf_ctx_run_mode method.
    Description: Test setting run_mode with valid and invalid values.
    Expectation: Valid run_mode is set, invalid run_mode raises ValueError.
    """
    Context.reset_instance()
    config_tpl = get_config_tpl()
    ctx = build_context(config_tpl)

    # Test valid run_mode
    ctx.set_mf_ctx_run_mode('train')
    assert ctx.mf_ctx_opr.run_mode == 'train'

    # Test invalid run_mode
    with pytest.raises(ValueError) as exc_info:
        ctx.set_mf_ctx_run_mode('invalid_mode')
    assert 'Invalid value' in str(exc_info.value)

    # Test None run_mode
    ctx.set_mf_ctx_run_mode(None)
    Context.reset_instance()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_ms_context_operator_set_save_graphs_path():
    """
    Feature: Test MSContextOperator._set_save_graphs_path.
    Description: Test save_graphs_path setting.
    Expectation: save_graphs_path is set when save_graphs is True.
    """
    config = MindFormerConfig(
        context={'save_graphs': True, 'save_graphs_path': '/tmp/graphs'},
        parallel={}
    )
    operator = MSContextOperator(config)
    assert operator.get_context('save_graphs_path') == '/tmp/graphs'


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_ms_context_operator_predict_jit_config_o1():
    """
    Feature: Test MSContextOperator._set_predict_jit_config with O1.
    Description: Test that O1 jit_level raises ValueError in predict mode.
    Expectation: ValueError is raised.
    """
    config = MindFormerConfig(
        run_mode='predict',
        context={'jit_level': 'O1'},
        parallel={}
    )
    with pytest.raises(ValueError) as exc_info:
        MSContextOperator(config)
    assert 'O1 is not supported' in str(exc_info.value)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_ms_context_operator_predict_jit_config_o2_with_boost():
    """
    Feature: Test MSContextOperator._set_predict_jit_config with O2 and boost.
    Description: Test that O2 with infer_boost=on raises ValueError.
    Expectation: ValueError is raised.
    """
    config = MindFormerConfig(
        run_mode='predict',
        context={'jit_level': 'O2', 'infer_boost': 'on'},
        parallel={}
    )
    with pytest.raises(ValueError) as exc_info:
        MSContextOperator(config)
    assert 'infer_boost must set off' in str(exc_info.value)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_ms_context_operator_predict_jit_config_o2_without_boost():
    """
    Feature: Test MSContextOperator._set_predict_jit_config with O2 without boost.
    Description: Test that O2 with infer_boost=off works.
    Expectation: jit_config is set correctly.
    """
    def is_ms_context_operator_predict_jit_config():
        config = MindFormerConfig(
            run_mode='predict',
            context={'jit_level': 'O2', 'infer_boost': 'off'},
            parallel={}
        )
        operator = MSContextOperator(config)
        assert operator.get_context("jit_level") == "O2"
        assert operator.get_context("infer_boost") == "off"
    run_in_subprocess(is_ms_context_operator_predict_jit_config)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_ms_context_operator_predict_jit_config_from_jit_config():
    """
    Feature: Test MSContextOperator._set_predict_jit_config with jit_config dict.
    Description: Test that jit_config dict is used.
    Expectation: jit_config values are taken from jit_config dict.
    """
    config = MindFormerConfig(
        run_mode='predict',
        context={
            'jit_level': 'O0',
            'infer_boost': 'on',
            'jit_config': {'jit_level': 'O2', 'infer_boost': 'off'}
        },
        parallel={}
    )
    operator = MSContextOperator(config)
    assert operator.get_context("jit_level") == "O2"
    assert operator.get_context("infer_boost") == "off"


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_set_context_without_build():
    """
    Feature: Test set_context without building context first.
    Description: Test that set_context raises RuntimeError when Context doesn't exist.
    Expectation: RuntimeError is raised.
    """
    Context.reset_instance()
    with pytest.raises(RuntimeError) as exc_info:
        set_context(run_mode='train')
    assert 'Build a Context instance' in str(exc_info.value)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_get_context_without_build():
    """
    Feature: Test get_context without building context first.
    Description: Test that get_context raises RuntimeError when Context doesn't exist.
    Expectation: RuntimeError is raised.
    """
    Context.reset_instance()
    with pytest.raises(RuntimeError) as exc_info:
        get_context('mode')
    assert 'Build a Context instance' in str(exc_info.value)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_build_parallel_context():
    """
    Feature: Test build_parallel_context function.
    Description: Test building parallel context.
    Expectation: ParallelOperator is returned.
    """
    config_tpl = get_config_tpl()
    parallel_opr = build_parallel_context(config_tpl)
    assert parallel_opr is not None
    assert hasattr(parallel_opr, 'parallel_ctx')
    assert hasattr(parallel_opr, 'parallel')


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@patch('mindformers.core.context.build_context.get_real_local_rank')
@patch('mindformers.core.context.build_context.ms.runtime.set_cpu_affinity')
def test_set_ms_affinity_with_affinity_config(mock_set_affinity, mock_rank):
    """
    Feature: Test set_ms_affinity with affinity_config.
    Description: Verify affinity_config overrides affinity_cpu_list and passes module config.
    Expectation: MindSpore set_cpu_affinity called with config values.
    """
    mock_rank.return_value = 1
    affinity_config = {
        'device_1': {
            'affinity_cpu_list': [0, 1],
            'module_to_cpu_dict': {'module_a': [2, 3]}
        }
    }
    set_ms_affinity(affinity_config, [4, 5])
    mock_set_affinity.assert_called_once_with(
        True,
        [0, 1],
        {'module_a': [2, 3]}
    )


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@patch('mindformers.core.context.build_context.get_real_local_rank')
@patch('mindformers.core.context.build_context.ms.runtime.set_cpu_affinity')
def test_set_ms_affinity_without_device_entry(mock_set_affinity, mock_rank):
    """
    Feature: Test set_ms_affinity when device entry missing.
    Description: Verify defaults are used when affinity_config lacks device info.
    Expectation: MindSpore set_cpu_affinity called with None values.
    """
    mock_rank.return_value = 0
    affinity_config = {
        'device_1': {
            'affinity_cpu_list': [4, 5],
            'module_to_cpu_dict': {'module_a': [6, 7]}
        }
    }
    set_ms_affinity(affinity_config, None)
    mock_set_affinity.assert_called_once_with(
        True,
        None,
        None
    )


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@patch('mindformers.core.context.build_context.get_cann_workqueue_cores', return_value=[0, 1])
@patch('mindformers.core.context.build_context.psutil.Process')
@patch('mindformers.core.context.build_context.psutil.cpu_count', return_value=8)
@patch('mindformers.core.context.build_context.ds.config.set_numa_enable')
def test_set_cpu_affinity_bind_available_cpus(mock_set_numa, mock_cpu_count,
                                              mock_process_cls, mock_get_cores,
                                              monkeypatch):
    """
    Feature: Test set_cpu_affinity binding behavior.
    Description: Verify CPU affinity excludes CANN workqueue cores when available.
    Expectation: Process cpu_affinity receives filtered CPU list.
    """
    monkeypatch.setenv('CPU_AFFINITY', 'True')
    process_mock = mock_process_cls.return_value

    set_cpu_affinity(rank_id=0, rank_size=2)

    mock_set_numa.assert_called_once_with(True)
    mock_cpu_count.assert_called_once()
    mock_get_cores.assert_called_once_with(0)
    process_mock.cpu_affinity.assert_called_once_with([2, 3])


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@patch('mindformers.core.context.build_context.get_cann_workqueue_cores', return_value=[0, 1, 2, 3])
@patch('mindformers.core.context.build_context.psutil.Process')
@patch('mindformers.core.context.build_context.psutil.cpu_count', return_value=8)
@patch('mindformers.core.context.build_context.ds.config.set_numa_enable')
def test_set_cpu_affinity_fallback_when_all_cores_taken(mock_set_numa, mock_cpu_count,
                                                        mock_process_cls, mock_get_cores,
                                                        monkeypatch):
    """
    Feature: Test set_cpu_affinity fallback behavior.
    Description: Verify original CPU list is used when CANN occupies all candidate cores.
    Expectation: Process cpu_affinity receives unfiltered CPU list.
    """
    monkeypatch.setenv('CPU_AFFINITY', 'True')
    process_mock = mock_process_cls.return_value

    set_cpu_affinity(rank_id=0, rank_size=2)

    mock_set_numa.assert_called_once_with(True)
    mock_cpu_count.assert_called_once()
    mock_get_cores.assert_called_once_with(0)
    process_mock.cpu_affinity.assert_called_once_with([0, 1, 2, 3])
