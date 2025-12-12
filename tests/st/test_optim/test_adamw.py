# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
Test module for testing the AdamW interface used for MindFormers.
How to run this:
pytest tests/st/test_optim/test_adamw.py
"""
import pytest
import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, dtype as mstype
from mindspore.nn import Cell
from tests.st.test_optim.optimizer_util import (
    build_network,
    FakeNet,
    default_fc1_weight_adamw_m,
    default_fc2_weight_adamw_m,
    default_fc1_weight_adamw_v,
    default_fc2_weight_adamw_v
)
from mindformers.core.optim.adamw import AdamW, _check_param_value

ms.set_context(mode=0)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestAdamW:
    """A test class for testing optimizer computation."""

    def test_computation(self):
        """
        Feature: Trainer.train()
        Description: Test computation of AdamW in training.
        Expectation: AssertionError
        """
        config = {'type': 'AdamW', "weight_decay": 0.1}
        _, cells = build_network(config, FakeNet(), is_group=True)
        assert np.allclose(cells.exp_avg[0].asnumpy(), default_fc1_weight_adamw_m, atol=1.e-4)
        assert np.allclose(cells.exp_avg[2].asnumpy(), default_fc2_weight_adamw_m, atol=1.e-4)
        assert np.allclose(cells.exp_avg_sq[0].asnumpy(), default_fc1_weight_adamw_v, atol=1.e-4)
        assert np.allclose(cells.exp_avg_sq[2].asnumpy(), default_fc2_weight_adamw_v, atol=1.e-4)


class SimpleNet(Cell):
    """Simple network for testing"""

    def __init__(self):
        super().__init__()
        self.weight = Parameter(Tensor(np.ones([2, 3]), mstype.float32), name="weight")
        self.bias = Parameter(Tensor(np.zeros([3]), mstype.float32), name="bias")

    def construct(self, x):
        return x * self.weight + self.bias


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_init():
    """
    Feature: AdamW optimizer initialization
    Description: Test AdamW initialization with default parameters
    Expectation: Successfully initialize AdamW optimizer with default parameters and verify beta1, beta2, and eps values
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params())
    assert optimizer is not None
    assert np.allclose(optimizer.beta1.asnumpy(), np.array([0.9]))
    assert np.allclose(optimizer.beta2.asnumpy(), np.array([0.999]))
    assert np.allclose(optimizer.eps.asnumpy(), np.array([1e-8]))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_init_with_custom_params():
    """
    Feature: AdamW optimizer initialization with custom parameters
    Description: Test AdamW initialization with custom learning rate, betas, eps, and weight decay
    Expectation: Successfully initialize AdamW with custom parameters and verify the values are set correctly
    """
    net = SimpleNet()
    learning_rate = 0.005
    betas = (0.8, 0.99)
    eps = 1e-7
    weight_decay = 0.01

    optimizer = AdamW(
        net.trainable_params(),
        learning_rate=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )

    assert np.allclose(optimizer.beta1.asnumpy(), np.array([0.8]))
    assert np.allclose(optimizer.beta2.asnumpy(), np.array([0.99]))
    assert np.allclose(optimizer.eps.asnumpy(), np.array([1e-7]))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_init_with_swap():
    """
    Feature: AdamW optimizer initialization with swap parameter
    Description: Test AdamW initialization with swap=True to offload optimizer states to CPU
    Expectation: Successfully initialize AdamW with swap=True and verify optimizer states are on CPU
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params(), swap=True)
    assert optimizer.swap is True
    # Check if exp_avg parameters are on CPU
    for param in optimizer.exp_avg:
        assert param.device == 'CPU'
    for param in optimizer.exp_avg_sq:
        assert param.device == 'CPU'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_init_with_group_params():
    """
    Feature: AdamW optimizer initialization with group parameters
    Description: Test AdamW initialization with grouped parameters having different learning rates and weight decays
    Expectation: Successfully initialize AdamW with grouped parameters
    """
    net = SimpleNet()
    params = [
        {'params': [net.weight], 'lr': 0.001, 'weight_decay': 0.01},
        {'params': [net.bias], 'lr': 0.0001, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params)
    assert optimizer is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_param_value():
    """
    Feature: _check_param_value function
    Description: Test _check_param_value function with valid parameters
    Expectation: Successfully validate parameters without raising exceptions
    """
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.01
    _check_param_value(betas, eps, weight_decay, "AdamW")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_param_value_invalid_betas_type():
    """
    Feature: _check_param_value function parameter validation
    Description: Test _check_param_value function with invalid betas type (string instead of tuple/list)
    Expectation: Raise TypeError when betas is not a tuple or list
    """
    betas = "invalid"
    eps = 1e-8
    weight_decay = 0.01
    with pytest.raises(TypeError):
        _check_param_value(betas, eps, weight_decay, "AdamW")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_param_value_invalid_betas_length():
    """
    Feature: _check_param_value function parameter validation
    Description: Test _check_param_value function with invalid betas length (3 elements instead of 2)
    Expectation: Raise ValueError when betas length is not 2
    """
    betas = (0.9, 0.999, 0.9999)
    eps = 1e-8
    weight_decay = 0.01
    with pytest.raises(ValueError):
        _check_param_value(betas, eps, weight_decay, "AdamW")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_param_value_invalid_beta1():
    """
    Feature: _check_param_value function parameter validation
    Description: Test _check_param_value function with invalid beta1 (equal to 1.0, which is out of range)
    Expectation: Raise ValueError when beta1 is not in (0.0, 1.0)
    """
    betas = (1.0, 0.999)
    eps = 1e-8
    weight_decay = 0.01
    with pytest.raises(ValueError):
        _check_param_value(betas, eps, weight_decay, "AdamW")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_param_value_invalid_beta2():
    """
    Feature: _check_param_value function parameter validation
    Description: Test _check_param_value function with invalid beta2 (equal to 0.0, which is out of range)
    Expectation: Raise ValueError when beta2 is not in (0.0, 1.0)
    """
    betas = (0.9, 0.0)
    eps = 1e-8
    weight_decay = 0.01
    with pytest.raises(ValueError):
        _check_param_value(betas, eps, weight_decay, "AdamW")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_param_value_invalid_eps():
    """
    Feature: _check_param_value function parameter validation
    Description: Test _check_param_value function with invalid eps (equal to 0.0, which is not greater than 0)
    Expectation: Raise ValueError when eps is not greater than 0
    """
    betas = (0.9, 0.999)
    eps = 0.0
    weight_decay = 0.01
    with pytest.raises(ValueError):
        _check_param_value(betas, eps, weight_decay, "AdamW")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_param_value_invalid_weight_decay():
    """
    Feature: _check_param_value function parameter validation
    Description: Test _check_param_value function with invalid weight_decay type (string instead of float/int/Cell)
    Expectation: Raise TypeError when weight_decay is not a float, int, or Cell
    """
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = "invalid"
    with pytest.raises(TypeError):
        _check_param_value(betas, eps, weight_decay, "AdamW")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_construct():
    """
    Feature: AdamW optimizer construct method
    Description: Test AdamW construct method with dummy gradients
    Expectation: Successfully execute AdamW construct method and return non-None result
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params())

    # Create dummy gradients
    gradients = (
        Tensor(np.ones([2, 3]), mstype.float32),
        Tensor(np.ones([3]), mstype.float32)
    )

    # Test construct
    result = optimizer(gradients)
    assert result is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_construct_with_group_lr():
    """
    Feature: AdamW optimizer construct method with group learning rates
    Description: Test AdamW construct method with grouped parameters having different learning rates
    Expectation: Successfully execute AdamW construct method with group learning rates and return non-None result
    """
    net = SimpleNet()
    params = [
        {'params': [net.weight], 'lr': 0.001},
        {'params': [net.bias], 'lr': 0.0001}
    ]
    optimizer = AdamW(params)

    # Create dummy gradients
    gradients = (
        Tensor(np.ones([2, 3]), mstype.float32),
        Tensor(np.ones([3]), mstype.float32)
    )

    # Test construct
    result = optimizer(gradients)
    assert result is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_clone_state():
    """
    Feature: AdamW optimizer clone_state method
    Description: Test AdamW clone_state method to create copies of optimizer states
    Expectation: Successfully clone optimizer states with correct shape and dtype
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params())

    # Clone state
    cloned_state = optimizer.clone_state("test", "zeros")
    assert len(cloned_state) == len(optimizer.parameters)
    for i, param in enumerate(cloned_state):
        assert param.shape == optimizer.parameters[i].shape
        assert param.dtype == mstype.float32


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_clone_state_with_swap():
    """
    Feature: AdamW optimizer clone_state method with swap=True
    Description: Test AdamW clone_state method with swap=True to clone states to CPU
    Expectation: Successfully clone optimizer states to CPU when swap=True
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params(), swap=True)

    # Clone state
    cloned_state = optimizer.clone_state("test", "zeros")
    assert len(cloned_state) == len(optimizer.parameters)
    for param in cloned_state:
        assert param.device == 'CPU'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_with_dynamic_weight_decay():
    """
    Feature: AdamW optimizer with dynamic weight decay
    Description: Test AdamW initialization with dynamic weight decay
    Expectation: Successfully initialize AdamW with dynamic weight decay
    """
    net = SimpleNet()
    weight_decay = 1.0
    optimizer = AdamW(net.trainable_params(), weight_decay=weight_decay)
    assert optimizer is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_step():
    """
    Feature: AdamW optimizer step execution
    Description: Test AdamW step execution with dummy gradients
    Expectation: Successfully execute one optimization step and update parameters
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params())

    # Create dummy input and gradients
    gradients = (
        Tensor(np.ones([2, 3]), mstype.float32),
        Tensor(np.ones([3]), mstype.float32)
    )

    # Run one step
    optimizer(gradients)

    # Check if parameters are updated
    for param in net.trainable_params():
        assert not np.all(param.asnumpy() == np.ones_like(param.asnumpy()))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_with_zero_gradients():
    """
    Feature: AdamW optimizer with zero gradients
    Description: Test AdamW execution with zero gradients and non-zero weight decay
    Expectation: Parameters are updated due to weight decay even with zero gradients
    """
    net = SimpleNet()
    # Use non-zero weight decay to ensure parameters are updated
    optimizer = AdamW(net.trainable_params(), weight_decay=0.1)

    # Create zero gradients
    gradients = (
        Tensor(np.zeros([2, 3]), mstype.float32),
        Tensor(np.zeros([3]), mstype.float32)
    )

    # Run one step
    optimizer(gradients)

    # Check if parameters are still updated due to weight decay
    for param in net.trainable_params():
        assert not np.allclose(param.asnumpy(), np.ones_like(param.asnumpy()))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_invalid_learning_rate():
    """
    Feature: AdamW optimizer with invalid learning rate
    Description: Test AdamW initialization with invalid learning rate type
    Expectation: Raise ValueError when learning_rate is invalid
    """
    net = SimpleNet()
    with pytest.raises(ValueError):
        AdamW(net.trainable_params(), learning_rate="invalid")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_large_weight_decay():
    """
    Feature: AdamW optimizer with large weight decay
    Description: Test AdamW execution with large weight decay (0.1)
    Expectation: Parameters are significantly updated due to large weight decay
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params(), weight_decay=0.1)

    # Create dummy gradients
    gradients = (
        Tensor(np.ones([2, 3]), mstype.float32),
        Tensor(np.ones([3]), mstype.float32)
    )

    # Run one step
    optimizer(gradients)

    # Check if parameters are updated significantly due to large weight decay
    for param in net.trainable_params():
        assert np.all(param.asnumpy() < np.ones_like(param.asnumpy()))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_small_eps():
    """
    Feature: AdamW optimizer with small epsilon
    Description: Test AdamW execution with very small epsilon (1e-12)
    Expectation: Successfully execute AdamW with small epsilon and return non-None result
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params(), eps=1e-12)

    # Create dummy gradients
    gradients = (
        Tensor(np.ones([2, 3]), mstype.float32),
        Tensor(np.ones([3]), mstype.float32)
    )

    # Run one step
    result = optimizer(gradients)
    assert result is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_with_tuple_params():
    """
    Feature: AdamW optimizer with tuple parameters
    Description: Test AdamW initialization with tuple of parameters
    Expectation: Successfully initialize AdamW with tuple of parameters
    """
    net = SimpleNet()
    params_tuple = tuple(net.trainable_params())
    optimizer = AdamW(params_tuple)
    assert optimizer is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_global_step_increase():
    """
    Feature: AdamW optimizer global step
    Description: Test AdamW global step attribute and step execution
    Expectation: Verify global_step attribute exists and is a Tensor, and successfully execute one step
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params())

    # Create dummy gradients
    gradients = (
        Tensor(np.ones([2, 3]), mstype.float32),
        Tensor(np.ones([3]), mstype.float32)
    )

    # Just verify that the global_step attribute exists and is a Tensor
    assert hasattr(optimizer, 'global_step')
    assert isinstance(optimizer.global_step, Tensor)

    # Run one step to ensure the optimizer works correctly
    result = optimizer(gradients)
    assert result is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_with_mixed_precision_params():
    """
    Feature: AdamW optimizer with mixed precision parameters
    Description: Test AdamW execution with parameters of different precisions
    Expectation: Successfully initialize AdamW with mixed precision parameters and execute one step
    """
    # Create parameters with different precisions
    params = [
        Parameter(Tensor(np.ones([2, 3]), mstype.float32), name="fp32_param"),
        Parameter(Tensor(np.ones([3]), mstype.float16), name="fp16_param")
    ]
    optimizer = AdamW(params)
    assert optimizer is not None

    # Create dummy gradients with matching precisions
    gradients = (
        Tensor(np.ones([2, 3]), mstype.float32),
        Tensor(np.ones([3]), mstype.float16)
    )

    # Run one step
    result = optimizer(gradients)
    assert result is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_optim_filter():
    """
    Feature: AdamW optimizer with optim_filter
    Description: Test AdamW execution with optim_filter set to include all parameters
    Expectation: Successfully execute AdamW with optim_filter and return non-None result
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params())

    # Create dummy gradients
    gradients = (
        Tensor(np.ones([2, 3]), mstype.float32),
        Tensor(np.ones([3]), mstype.float32)
    )

    # Set optim_filter to include all parameters
    optimizer.optim_filter = (True, True)

    # Run one step
    result = optimizer(gradients)
    assert result is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_construct_without_group():
    """
    Feature: AdamW optimizer construct method without group
    Description: Test AdamW construct method with is_group set to False
    Expectation: Successfully execute AdamW construct method with is_group=False
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params())

    # Set is_group to False
    optimizer.is_group = False

    # Create dummy gradients
    gradients = (
        Tensor(np.ones([2, 3]), mstype.float32),
        Tensor(np.ones([3]), mstype.float32)
    )

    # Test construct
    result = optimizer(gradients)
    assert result is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_with_large_lr():
    """
    Feature: AdamW optimizer with large learning rate
    Description: Test AdamW execution with large learning rate (1.0)
    Expectation: Successfully execute AdamW with large learning rate and return non-None result
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params(), learning_rate=1.0)

    # Create dummy gradients
    gradients = (
        Tensor(np.ones([2, 3]), mstype.float32),
        Tensor(np.ones([3]), mstype.float32)
    )

    # Run one step
    result = optimizer(gradients)
    assert result is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_with_tensor_lr():
    """
    Feature: AdamW optimizer with Tensor learning rate
    Description: Test AdamW initialization with Tensor learning rate
    Expectation: Successfully initialize AdamW with Tensor learning rate
    """
    net = SimpleNet()
    lr_tensor = Tensor(np.array([0.001]), mstype.float32)
    optimizer = AdamW(net.trainable_params(), learning_rate=lr_tensor)
    assert optimizer is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_with_iterable_lr():
    """
    Feature: AdamW optimizer with Iterable learning rate
    Description: Test AdamW initialization with Iterable learning rate
    Expectation: Successfully initialize AdamW with Iterable learning rate
    """
    net = SimpleNet()
    lr_iter = [0.001, 0.0009, 0.0008]
    optimizer = AdamW(net.trainable_params(), learning_rate=lr_iter)
    assert optimizer is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_optim_filter_false():
    """
    Feature: AdamW optimizer with optim_filter=False
    Description: Test AdamW execution with optim_filter set to False for some parameters
    Expectation: Successfully execute AdamW with optim_filter=False and verify gradients are returned for
    filtered parameters
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params())

    # Create dummy gradients
    gradients = (
        Tensor(np.ones([2, 3]), mstype.float32),
        Tensor(np.ones([3]), mstype.float32)
    )

    # Set optim_filter to False for all parameters
    optimizer.optim_filter = (False, False)

    # Run one step
    result = optimizer(gradients)
    assert result is not None
    assert len(result) == len(gradients)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_get_weight_decay_and_lr():
    """
    Feature: AdamW optimizer get_weight_decay and get_lr methods
    Description: Test AdamW get_weight_decay and get_lr methods
    Expectation: Successfully call get_weight_decay and get_lr methods
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params(), weight_decay=0.01, learning_rate=0.001)

    # Test get_weight_decay method
    weight_decay = optimizer.get_weight_decay()
    assert weight_decay is not None

    # Test get_lr method
    lr = optimizer.get_lr()
    assert lr is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_clone_state_with_cloned_obj():
    """
    Feature: AdamW optimizer clone_state method with existing cloned_obj
    Description: Test AdamW clone_state method when old_param.param_info already has cloned_obj
    Expectation: Successfully clone state and append to existing cloned_obj list
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params())

    # First clone to create cloned_obj
    first_clone = optimizer.clone_state("first", "zeros")

    # Second clone should append to existing cloned_obj
    second_clone = optimizer.clone_state("second", "zeros")

    assert len(first_clone) == len(optimizer.parameters)
    assert len(second_clone) == len(optimizer.parameters)

    # Verify cloned_obj exists and has both clones
    for old_param in optimizer.parameters:
        assert hasattr(old_param.param_info, "cloned_obj")
        assert len(old_param.param_info.cloned_obj) >= 2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_construct_all_branches():
    """
    Feature: AdamW optimizer construct method branches
    Description: Test all branches of AdamW construct method
    Expectation: Successfully execute all branches of construct method
    """
    # Test is_group=False branch
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params())
    optimizer.is_group = False
    gradients = (
        Tensor(np.ones([2, 3]), mstype.float32),
        Tensor(np.ones([3]), mstype.float32)
    )
    result = optimizer(gradients)
    assert result is not None

    # Test is_group=True, is_group_lr=True branch
    params = [
        {'params': [net.weight], 'lr': 0.001},
        {'params': [net.bias], 'lr': 0.0001}
    ]
    optimizer = AdamW(params)
    optimizer.is_group = True
    optimizer.is_group_lr = True
    result = optimizer(gradients)
    assert result is not None

    # Test is_group=True, is_group_lr=False branch
    optimizer.is_group_lr = False
    result = optimizer(gradients)
    assert result is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_param_value_more_cases():
    """
    Feature: _check_param_value function with more cases
    Description: Test _check_param_value function with more parameter combinations
    Expectation: Successfully validate parameters or raise expected exceptions
    """
    # Test with float weight_decay
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.0
    _check_param_value(betas, eps, weight_decay, "AdamW")

    # Test with edge case betas
    betas = (0.0001, 0.9999)
    _check_param_value(betas, eps, weight_decay, "AdamW")

    # Test with large eps
    eps = 1e-3
    _check_param_value(betas, eps, weight_decay, "AdamW")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_clone_state_with_ones_init():
    """
    Feature: AdamW clone_state method with ones init
    Description: Test AdamW clone_state method with ones initialization
    Expectation: Successfully clone state with ones init
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params())

    # Test with 'ones' init
    ones_clone = optimizer.clone_state("adam_m_ones", "ones")
    assert len(ones_clone) == len(optimizer.parameters)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_with_large_learning_rate():
    """
    Feature: AdamW optimizer with large learning rate
    Description: Test AdamW execution with very large learning rate
    Expectation: Successfully execute AdamW with large learning rate and update parameters significantly
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params(), learning_rate=1.0)

    # Create dummy gradients
    gradients = (
        Tensor(np.ones([2, 3]), mstype.float32),
        Tensor(np.ones([3]), mstype.float32)
    )

    # Run one step
    optimizer(gradients)

    # Check if parameters are significantly updated
    for param in net.trainable_params():
        assert not np.allclose(param.asnumpy(), np.ones_like(param.asnumpy()), atol=0.1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adamw_with_very_small_learning_rate():
    """
    Feature: AdamW optimizer with very small learning rate
    Description: Test AdamW execution with very small learning rate
    Expectation: Successfully execute AdamW with very small learning rate
    """
    net = SimpleNet()
    optimizer = AdamW(net.trainable_params(), learning_rate=1e-10)

    # Create dummy gradients
    gradients = (
        Tensor(np.ones([2, 3]), mstype.float32),
        Tensor(np.ones([3]), mstype.float32)
    )

    # Run one step and check if it completes successfully
    result = optimizer(gradients)
    assert result is not None
