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
"""
test parameter_register
"""
import unittest
import mindspore as ms
from mindspore import Tensor
import pytest
import numpy as np

from mindformers.utils.parameter_register import ParameterRegister


class TestParameterRegister(unittest.TestCase):
    """Test cases for ParameterRegister class."""

    def setUp(self):
        """Set up test environment."""
        ParameterRegister._instance = None  # pylint: disable=protected-access

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_singleton_pattern(self):
        """
        Test ParameterRegister singleton pattern implementation.
        Input: Multiple calls to ParameterRegister().
        Output: Same instance returned.
        Expected: instance1 is instance2.
        """
        instance1 = ParameterRegister()
        instance2 = ParameterRegister()
        self.assertIs(instance1, instance2)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_register_new_parameter_with_tensor(self):
        """
        Test registering new parameter with Tensor value.
        Input: Parameter name "test_param" and Tensor([1.0, 2.0, 3.0], ms.float32).
        Output: Registered Parameter object.
        Expected: Parameter has correct name, requires_grad=False, and matching values.
        """
        register = ParameterRegister()
        tensor_value = Tensor([1.0, 2.0, 3.0], ms.float32)
        param = register.register("test_param", tensor_value)
        self.assertEqual(param.name, "test_param")
        self.assertFalse(param.requires_grad)
        np.testing.assert_array_equal(param.asnumpy(), tensor_value.asnumpy())

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_register_new_parameter_with_list(self):
        """
        Test registering new parameter with list value.
        Input: Parameter name "test_param" and [1.0, 2.0, 3.0] list.
        Output: Registered Parameter object.
        Expected: Parameter has correct name and values converted from list to Tensor.
        """
        register = ParameterRegister()
        list_value = [1.0, 2.0, 3.0]
        param = register.register("test_param", list_value)
        self.assertEqual(param.name, "test_param")
        expected_tensor = Tensor(list_value, ms.float32)
        np.testing.assert_array_equal(param.asnumpy(), expected_tensor.asnumpy())

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_register_parameter_with_options(self):
        """
        Test registering parameter with additional options.
        Input: Parameter name "test_param", Tensor([1.0, 2.0, 3.0], ms.float32), requires_grad=True,
            parallel_optimizer=True.
        Output: Registered Parameter object.
        Expected: Parameter has correct name, requires_grad=True, and parallel_optimizer=True.
        """
        register = ParameterRegister()
        tensor_value = Tensor([1.0, 2.0, 3.0], ms.float32)
        param = register.register(
            "test_param", tensor_value, requires_grad=True, parallel_optimizer=True
        )
        self.assertEqual(param.name, "test_param")
        self.assertTrue(param.requires_grad)
        self.assertTrue(param.parallel_optimizer)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_register_duplicate_parameter(self):
        """
        Test registering duplicate parameter name raises exception.
        Input: Register "test_param" twice with different values.
        Output: KeyError exception.
        Expected: Exception message contains "Parameter 'test_param' already registered.".
        """
        register = ParameterRegister()
        tensor_value1 = Tensor([1.0, 2.0, 3.0], ms.float32)
        tensor_value2 = Tensor([4.0, 5.0, 6.0], ms.float32)
        register.register("test_param", tensor_value1)
        with self.assertRaises(KeyError) as context:
            register.register("test_param", tensor_value2)
        self.assertIn("Parameter 'test_param' already registered.", str(context.exception))

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_get_existing_parameter(self):
        """
        Test getting existing registered parameter.
        Input: Register "test_param" then get it.
        Output: Retrieved Parameter object.
        Expected: Retrieved parameter is the same object as registered one.
        """
        register = ParameterRegister()
        tensor_value = Tensor([1.0, 2.0, 3.0], ms.float32)
        registered_param = register.register("test_param", tensor_value)
        retrieved_param = register.get("test_param")
        self.assertIs(registered_param, retrieved_param)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_get_nonexistent_parameter(self):
        """
        Test getting non-existent parameter raises exception.
        Input: Get "nonexistent_param" without registering it.
        Output: KeyError exception.
        Expected: Exception message contains "nonexistent_param not registered.".
        """
        register = ParameterRegister()
        with self.assertRaises(KeyError) as context:
            register.get("nonexistent_param")
        self.assertIn("nonexistent_param not registered.", str(context.exception))

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_clear_existing_parameter(self):
        """
        Test clearing existing registered parameter.
        Input: Register "test_param" then clear it.
        Output: Parameter values set to zero.
        Expected: Parameter data equals Tensor([0.0, 0.0, 0.0], ms.float32).
        """
        register = ParameterRegister()
        tensor_value = Tensor([1.0, 2.0, 3.0], ms.float32)
        param = register.register("test_param", tensor_value)
        register.clear("test_param")
        expected_zeros = Tensor([0.0, 0.0, 0.0], ms.float32)
        np.testing.assert_array_equal(param.asnumpy(), expected_zeros.asnumpy())

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_clear_nonexistent_parameter(self):
        """
        Test clearing non-existent parameter raises exception.
        Input: Clear "nonexistent_param" without registering it.
        Output: KeyError exception.
        Expected: Exception message contains "nonexistent_param not registered.".
        """
        register = ParameterRegister()
        with self.assertRaises(KeyError) as context:
            register.clear("nonexistent_param")
        self.assertIn("nonexistent_param not registered.", str(context.exception))

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_items_keys_values(self):
        """
        Test items, keys, values methods.
        Input: Register two parameters "param1" and "param2".
        Output: Results from keys(), values(), and items() methods.
        Expected: Keys contain both parameter names, values contain both parameters,
                 and items return key-value pairs matching registered parameters.
        """
        register = ParameterRegister()
        tensor_value1 = Tensor([1.0, 2.0, 3.0], ms.float32)
        tensor_value2 = Tensor([4.0, 5.0, 6.0], ms.float32)
        register.register("param1", tensor_value1)
        register.register("param2", tensor_value2)
        keys = list(register.keys())
        self.assertIn("param1", keys)
        self.assertIn("param2", keys)
        self.assertEqual(len(keys), 2)
        values = list(register.values())
        self.assertEqual(len(values), 2)
        for k, v in register.items():
            assert k in keys
            assert v is register.get(k)
