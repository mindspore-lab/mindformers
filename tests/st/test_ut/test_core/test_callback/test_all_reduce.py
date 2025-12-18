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
"""Test test_all_reduce.py"""
from unittest.mock import patch, MagicMock, Mock
import unittest
import pytest
import numpy as np
from mindspore import Tensor
import mindformers.core.callback.callback as callback_module


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions in callback.py"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_loss_output_with_tuple(self):
        """Test _get_loss_output function handling tuple output"""
        # Prepare test data
        loss_tensor = Tensor(np.array([0.5]))
        overflow_tensor = Tensor(np.array([False]))
        scaling_sens_tensor = Tensor(np.array([1.0]))
        lr_tensor = Tensor(np.array([0.01]))
        norm_tensor = Tensor(np.array([1.0]))

        output = (loss_tensor, overflow_tensor, scaling_sens_tensor, lr_tensor, norm_tensor)

        # Execute test
        # pylint: disable=W0212
        loss, overflow, scaling_sens, learning_rate, global_norm = callback_module._get_loss_output(output)

        # Verify results
        self.assertEqual(loss, 0.5)
        self.assertFalse(overflow)
        self.assertEqual(scaling_sens, 1.0)
        self.assertEqual(learning_rate, 0.01)
        self.assertEqual(global_norm, 1.0)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_loss_output_with_single_tensor(self):
        """Test _get_loss_output function handling single tensor"""
        loss_tensor = Tensor(np.array([0.8]))
        output = loss_tensor

        # pylint: disable=W0212
        loss, overflow, _, _, _ = callback_module._get_loss_output(output)

        self.assertEqual(loss, 0.8)
        self.assertFalse(overflow)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.core.callback.callback.P')
    def test_init(self, mock_p):
        """Test AllReduceNet initialization"""
        # Mock AllReduce operation
        mock_allreduce = MagicMock()
        mock_p.AllReduce.return_value = mock_allreduce
        mock_p.ReduceOp.SUM = "SUM"

        # Create instance
        net = callback_module.AllReduceNet("test_group")

        # Verify initialization
        mock_p.AllReduce.assert_called_once_with(op="SUM", group="test_group")
        self.assertTrue(net.get_flags()['skip_auto_parallel_compile'])

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.core.callback.callback.P')
    def test_construct(self, mock_p):
        """Test construct method"""
        # Mock AllReduce operation
        mock_allreduce = MagicMock()
        mock_allreduce.return_value = Tensor([2.0])
        mock_p.AllReduce.return_value = mock_allreduce
        mock_p.ReduceOp.SUM = "SUM"

        net = callback_module.AllReduceNet("test_group")
        input_tensor = Tensor([1.0])

        # Execute construct
        result = net.construct(input_tensor)

        # Verify call
        mock_allreduce.assert_called_once_with(input_tensor)
        self.assertEqual(result.asnumpy(), np.array([2.0]))

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_optimizer_state(self):
        """Test _get_optimizer_state function"""
        # Create mock parameters
        param1 = Mock()
        param1.name = "weight1"
        param1.to.return_value = param1
        param1.norm.return_value = Tensor(np.array([1.0]))

        param2 = Mock()
        param2.name = "bias1"
        param2.to.return_value = param2
        param2.norm.return_value = Tensor(np.array([0.5]))

        optim_params = [param1, param2]

        # pylint: disable=W0212
        norms = callback_module._get_optimizer_state(optim_params)

        self.assertIn("weight1", norms)
        self.assertIn("bias1", norms)
        self.assertEqual(norms["weight1"], 1.0)
        self.assertEqual(norms["bias1"], 0.5)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.core.callback.callback.parameter_register')
    def test_get_separate_loss(self, mock_parameter_register):
        """Test getting separate loss values"""
        # Mock parameter register return values
        mock_aux_loss = MagicMock()
        mock_aux_loss.asnumpy.return_value = np.array(0.1)
        mock_mtp_loss = MagicMock()
        mock_mtp_loss.asnumpy.return_value = np.array(0.2)
        mock_lm_loss = MagicMock()
        mock_lm_loss.asnumpy.return_value = np.array(0.3)

        mock_parameter_register.get.side_effect = lambda x: {
            "aux_loss": mock_aux_loss,
            "mtp_loss": mock_mtp_loss,
            "lm_loss": mock_lm_loss
        }[x]

        # Mock clear method
        mock_parameter_register.clear = MagicMock()

        # Call function
        # pylint: disable=W0212
        lm_loss, aux_loss, mtp_loss = callback_module._get_separate_loss()

        # Verify return values
        self.assertEqual(lm_loss, np.array(0.3))
        self.assertEqual(aux_loss, np.array(0.1))
        self.assertEqual(mtp_loss, np.array(0.2))

        # Verify clear method was called
        mock_parameter_register.clear.assert_any_call("aux_loss")
        mock_parameter_register.clear.assert_any_call("mtp_loss")
        mock_parameter_register.clear.assert_any_call("lm_loss")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.core.callback.callback.logger')
    @patch('mindformers.core.callback.callback.ms.ops.randn')
    def test_zero_norm_vector(self, mock_randn, mock_logger):
        """Test zero norm vector case"""
        # Mock zero norm vector
        mock_u_tensor = MagicMock()
        mock_u_tensor.norm.return_value = Tensor(np.array(0.0))
        mock_randn.return_value = mock_u_tensor

        # Execute test
        # pylint: disable=W0212
        result = callback_module._get_max_eigenvalue(Tensor(np.array([[1.0, 0.0], [0.0, 1.0]])), 5)

        # Verify return value is 0
        self.assertEqual(result, 0.0)

        # Verify warning log
        mock_logger.warning.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.core.callback.callback._get_max_eigenvalue')
    @patch('mindformers.core.callback.callback.ms.ops.norm')
    @patch('mindformers.core.callback.callback.ms.ops.square')
    def test_get_stable_rank_success(self, mock_square, mock_norm, mock_get_max_eigenvalue):
        """Test successful stable rank calculation"""
        # Mock parameter
        mock_weight = MagicMock()
        mock_weight.name = "test_weight"

        # Mock eigenvalue calculation - return Tensor instead of numpy array
        mock_get_max_eigenvalue.return_value = Tensor([2.0])

        # Mock Frobenius norm calculation - return Tensor
        mock_norm.return_value = Tensor([4.0])

        # Mock square calculation - return Tensor
        mock_square.return_value = Tensor([16.0])

        # Execute test
        # pylint: disable=W0212
        stable_rank, eigenvalue = callback_module._get_stable_rank(mock_weight, 5)

        # Verify results
        self.assertEqual(stable_rank, 8.0)  # 16.0 / 2.0
        self.assertEqual(eigenvalue, 2.0)

        # Verify function calls
        mock_get_max_eigenvalue.assert_called_once_with(mock_weight, 5)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.core.callback.callback._get_max_eigenvalue')
    @patch('mindformers.core.callback.callback.logger')
    def test_get_stable_rank_exception(self, mock_logger, mock_get_max_eigenvalue):
        """Test calculation exception case"""
        # Mock parameter
        mock_weight = MagicMock()
        mock_weight.name = "test_weight"

        # Mock exception
        mock_get_max_eigenvalue.side_effect = Exception("Calculation error")

        # Execute test
        # pylint: disable=W0212
        stable_rank, eigenvalue = callback_module._get_stable_rank(mock_weight, 5)

        # Verify return values
        self.assertEqual(stable_rank, 0.0)
        self.assertEqual(eigenvalue, 0.0)

        # Verify warning log
        mock_logger.warning.assert_called_once()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.core.callback.callback._get_max_eigenvalue')
    @patch('mindformers.core.callback.callback.ms.ops.norm')
    @patch('mindformers.core.callback.callback.ms.ops.square')
    def test_zero_eigenvalue(self, mock_square, mock_norm, mock_get_max_eigenvalue):
        """Test zero eigenvalue case"""
        # Mock parameter
        mock_weight = MagicMock()
        mock_weight.name = "test_weight"

        # Mock zero eigenvalue - return Tensor instead of numpy array
        mock_get_max_eigenvalue.return_value = Tensor([0.0])

        # Mock Frobenius norm calculation - return Tensor
        mock_norm.return_value = Tensor([4.0])

        # Mock square calculation - return Tensor
        mock_square.return_value = Tensor([16.0])

        # Execute test
        # pylint: disable=W0212
        stable_rank, eigenvalue = callback_module._get_stable_rank(mock_weight, 5)

        # Verify return values
        self.assertEqual(stable_rank, 0.0)
        self.assertEqual(eigenvalue, 0.0)


class TestAllReduceNet(unittest.TestCase):
    """Test AllReduceNet class"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_all_reduce_net_init(self):
        """Test AllReduceNet initialization"""
        net = callback_module.AllReduceNet("test_group")

        # Verify skip_auto_parallel_compile flag is set
        self.assertTrue(net.get_flags()['skip_auto_parallel_compile'])


if __name__ == '__main__':
    unittest.main()
