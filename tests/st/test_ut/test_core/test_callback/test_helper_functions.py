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
"""Test callback.py using pytest framework."""
import inspect
from unittest.mock import Mock, patch

import numpy as np
import pytest

from mindformers.core.callback.callback import (
    AllReduceNet,
    _check_mspti_is_on,
    _get_loss_output,
    _get_max_eigenvalue,
    _get_optimizer_state,
    _get_separate_loss,
    _get_stable_rank,
    _get_weight_norm,
    _log_grouped_lr_info,
    get_embedding_info
)

# pylint: disable=unused-argument   # for mock logic


class TestHelperFunctions:
    """Test helper functions in callback.py"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_loss_output(self):
        """Test _get_loss_output function."""
        # Test case 1: Simple scalar output (not Tensor)
        output = 0.5
        loss, overflow, scaling_sens, _, _ = _get_loss_output(output)
        assert loss == 0.5
        assert not overflow
        assert not scaling_sens

        # Test case 2: Tuple with 3 elements
        output = (0.5, False, 1024.0)
        loss, overflow, scaling_sens, _, _ = _get_loss_output(output)
        assert loss == 0.5
        assert not overflow
        assert scaling_sens == 1024.0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.F')
    def test_get_weight_norm(self, mock_f):
        """Test _get_weight_norm function."""
        network = Mock()
        param = Mock()
        param.to.return_value.norm.return_value = 1.0
        network.trainable_params.return_value = [param, param]

        # Mock F.stack
        mock_f.stack.return_value.norm.return_value.item.return_value = 1.414

        norm = _get_weight_norm(network)
        assert norm == pytest.approx(1.414)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_optimizer_state(self):
        """Test _get_optimizer_state function"""
        param1 = Mock()
        param1.name = "p1"
        param1.to.return_value.norm.return_value.item.return_value = 0.1

        param2 = Mock()
        param2.name = "p2"
        param2.to.return_value.norm.return_value.item.return_value = 0.2

        optim_params = [param1, param2]

        norms = _get_optimizer_state(optim_params)
        assert norms['p1'] == 0.1
        assert norms['p2'] == 0.2


class TestAllReduceNet:
    """Test AllReduceNet class"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_init_and_construct(self):
        """Test AllReduceNet initialization"""

        # Mock P.AllReduce which is used in AllReduceNet.__init__
        mock_allreduce_class = Mock()
        mock_allreduce_instance = Mock()
        mock_allreduce_class.return_value = mock_allreduce_instance

        with patch('mindformers.core.callback.callback.P.AllReduce', mock_allreduce_class):
            net = AllReduceNet('test_group')
            mock_allreduce_class.assert_called_once()

            # Test construct method
            mock_tensor = Mock()
            mock_allreduce_instance.return_value = mock_tensor
            result = net.construct(mock_tensor)
            assert result == mock_tensor


class TestCheckMsptiIsOn:
    """Test _check_mspti_is_on function"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('os.getenv')
    def test_mspti_enabled(self, mock_getenv):
        """Test when libmspti.so is in LD_PRELOAD"""

        mock_getenv.return_value = "/path/to/libmspti.so"
        result = _check_mspti_is_on()
        assert result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('os.getenv')
    def test_mspti_disabled(self, mock_getenv):
        """Test when libmspti.so is not in LD_PRELOAD"""

        mock_getenv.return_value = "/path/to/other.so"
        result = _check_mspti_is_on()
        assert not result

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('os.getenv')
    def test_mspti_no_ld_preload(self, mock_getenv):
        """Test when LD_PRELOAD is not set"""

        mock_getenv.return_value = None
        result = _check_mspti_is_on()
        assert not result


class TestGetSeparateLoss:
    """Test _get_separate_loss function"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.parameter_register')
    def test_get_separate_loss(self, mock_param_register):
        """Test _get_separate_loss retrieves and clears losses"""

        # Mock parameter values
        mock_aux_loss = Mock()
        mock_aux_loss.asnumpy.return_value = np.array([0.1])
        mock_mtp_loss = Mock()
        mock_mtp_loss.asnumpy.return_value = np.array([0.2])
        mock_lm_loss = Mock()
        mock_lm_loss.asnumpy.return_value = np.array([0.3])

        mock_param_register.get.side_effect = lambda x, default=None: {
            'aux_loss': mock_aux_loss,
            'mtp_loss': mock_mtp_loss,
            'lm_loss': mock_lm_loss
        }.get(x, default)

        lm_loss, aux_loss, mtp_loss = _get_separate_loss()

        assert lm_loss[0] == 0.3
        assert aux_loss[0] == 0.1
        assert mtp_loss[0] == 0.2

        # Verify clear was called
        assert mock_param_register.clear.call_count == 3


class TestLogGroupedLrInfo:
    """Test _log_grouped_lr_info function"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_log_grouped_lr_info_basic(self):
        """Test _log_grouped_lr_info basic functionality"""
        # This test verifies the function can be called without errors
        # when GROUPED_PARAMS is empty (default state in our mocks)

        # Should return early without error when GROUPED_PARAMS is empty
        # If this raises an exception, pytest will fail the test
        _log_grouped_lr_info()


class TestGetLossOutputExtended:
    """Extended tests for _get_loss_output function"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_loss_output_tuple_4(self):
        """Test _get_loss_output with 4-element tuple"""

        output = (0.5, False, 1024.0, 0.001)
        loss, overflow, scaling_sens, learning_rate, global_norm = _get_loss_output(output)
        assert loss == 0.5
        assert not overflow
        assert scaling_sens == 1024.0
        assert learning_rate == 0.001
        assert global_norm is None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_loss_output_tuple_7(self):
        """Test _get_loss_output with 7-element tuple"""

        output = (0.5, False, 1024.0, 0.001, 2.5, np.array([1.0, 2.0]), 2)
        loss, overflow, scaling_sens, learning_rate, global_norm = _get_loss_output(output)
        assert loss == 0.5
        assert not overflow
        assert scaling_sens == 1024.0
        assert learning_rate == 0.001
        assert global_norm == 2.5


class TestGetMaxEigenvalue:
    """Test _get_max_eigenvalue function"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_max_eigenvalue_basic(self):
        """Test _get_max_eigenvalue function - simplified test"""
        # This function is complex and involves many MindSpore operations
        # We'll just verify it exists and has the correct signature

        # Verify the function exists
        assert callable(_get_max_eigenvalue)

        # Verify the function signature
        sig = inspect.signature(_get_max_eigenvalue)
        params = list(sig.parameters.keys())
        assert 'input_tensor' in params
        assert 'num_iter' in params

        # Note: Full functional testing of this method would require actual MindSpore tensors
        # which is beyond the scope of unit testing with mocks


class TestGetStableRankExtended:
    """Extended tests for _get_stable_rank function"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.ms.ops.square')
    @patch('mindformers.core.callback.callback.ms.ops.norm')
    @patch('mindformers.core.callback.callback._get_max_eigenvalue')
    def test_get_stable_rank_zero_eigenvalue(self, mock_eigenvalue, mock_norm, mock_square):
        """Test _get_stable_rank when eigenvalue is zero"""

        # Create a more complete mock weight object
        weight = Mock()
        weight.name = "test_weight"
        weight.ndim = 2  # 添加 ndim 属性，避免 -ndim 操作
        weight.shape = [3, 3]  # 添加 shape 属性

        mock_eigenvalue.return_value = np.array(0.0)

        stable_rank, eig = _get_stable_rank(weight, num_iter=5)
        assert stable_rank == 0.0
        assert eig == 0.0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.ms.ops.square')
    @patch('mindformers.core.callback.callback.ms.ops.norm')
    @patch('mindformers.core.callback.callback._get_max_eigenvalue')
    def test_get_stable_rank_normal(self, mock_eigenvalue, mock_norm, mock_square):
        """Test _get_stable_rank with normal values"""

        # Create a more complete mock weight object
        weight = Mock()
        weight.name = "test_weight"
        weight.ndim = 2  # 添加 ndim 属性，避免 -ndim 操作
        weight.shape = [3, 3]  # 添加 shape 属性

        mock_eigenvalue.return_value = np.array(2.0)

        # Mock norm to return a Mock that can be squared
        mock_norm_tensor = Mock()
        mock_norm_tensor.ndim = 0  # 标量
        mock_norm.return_value = mock_norm_tensor

        # Mock square to return a Mock that has asnumpy method returning 16.0
        mock_square_result = Mock()
        mock_square_result.asnumpy.return_value = 16.0
        mock_square.return_value = mock_square_result

        stable_rank, eig = _get_stable_rank(weight, num_iter=5)
        # stable_rank = f_norm^2 / eig = 16.0 / 2.0 = 8.0
        assert stable_rank == 8.0
        assert eig == 2.0


class TestGetEmbeddingInfo:
    """Test get_embedding_info function"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.get_group_size', return_value=8)
    @patch('mindformers.core.callback.callback.get_rank', return_value=0)
    @patch('mindspore.context.get_auto_parallel_context', return_value=2)
    def test_get_embedding_info(self, *mocks):
        """Test get_embedding_info extracts embedding local norm"""

        cb_params = Mock()
        cb_params.net_outputs = [0.5, False, 1024.0, 0.001, 2.5,
                                 [1.0, 2.0, 3.0], [128, 256, 128]]

        embedding_size = 128
        result = get_embedding_info(cb_params, embedding_size)

        # Should return the first local_norm with matching size
        assert result == 1.0


class TestGetMaxEigenvalueComprehensive:
    """Comprehensive tests for _get_max_eigenvalue function"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.ms.ops.matmul')
    @patch('mindformers.core.callback.callback.ms.ops.unsqueeze')
    @patch('mindformers.core.callback.callback.ms.ops.randn')
    @patch('mindformers.core.callback.callback.logger')
    def test_get_max_eigenvalue_2d_tensor(self, mock_logger, mock_randn,
                                          mock_unsqueeze, mock_matmul):
        """Test _get_max_eigenvalue with 2D tensor"""

        # Create mock input tensor
        input_tensor = Mock()
        input_tensor.ndim = 2
        input_tensor.shape = [3, 3]
        input_tensor.astype.return_value = input_tensor
        input_tensor.transpose.return_value = input_tensor

        # Mock randn to return a tensor with positive norm
        mock_u_tensor = Mock()
        mock_u_norm = Mock()
        mock_u_norm.asnumpy.return_value = 1.0
        mock_u_tensor.norm.return_value = mock_u_norm
        mock_u_tensor.__truediv__ = Mock(return_value=mock_u_tensor)
        mock_randn.return_value = mock_u_tensor

        # Mock unsqueeze
        mock_unsqueeze.return_value = mock_u_tensor

        # Mock matmul operations
        mock_input_seq = Mock()
        mock_unsqueeze.return_value = mock_input_seq

        mock_v_tensor = Mock()
        mock_v_norm = Mock()
        mock_v_norm.asnumpy.return_value = 1.0

        # Mock (v_norm != 0).all() - need to return a tensor-like object with .all() method
        mock_comparison_result = Mock()
        mock_comparison_result.all.return_value = True
        mock_v_norm.__ne__ = Mock(return_value=mock_comparison_result)

        mock_v_tensor.norm.return_value = mock_v_norm
        mock_v_tensor.transpose.return_value = mock_v_tensor
        mock_v_tensor.__truediv__ = Mock(return_value=mock_v_tensor)

        mock_eigenvalue = Mock()
        mock_eigenvalue.asnumpy.return_value = 2.5
        mock_eigenvalue.squeeze.return_value = mock_eigenvalue

        # matmul is called:
        # 1. Once for input_seq (line 211)
        # 2. num_iter times for v_tensor (line 216)
        # 3. num_iter times for eigenvalue (line 217)
        # Total: 1 + 2 + 2 = 5 times for num_iter=2
        mock_matmul.side_effect = [
            mock_input_seq,  # Line 211: input_seq calculation
            mock_v_tensor,  # Line 216: iteration 1, v_tensor
            mock_eigenvalue,  # Line 217: iteration 1, eigenvalue
            mock_v_tensor,  # Line 216: iteration 2, v_tensor
            mock_eigenvalue  # Line 217: iteration 2, eigenvalue
        ]

        result = _get_max_eigenvalue(input_tensor, num_iter=2)
        assert result == 2.5

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @patch('mindformers.core.callback.callback.ms.ops.randn')
    @patch('mindformers.core.callback.callback.logger')
    def test_get_max_eigenvalue_zero_norm(self, mock_logger, mock_randn):
        """Test _get_max_eigenvalue when random vector has zero norm"""

        input_tensor = Mock()
        input_tensor.ndim = 2
        input_tensor.shape = [3, 3]
        input_tensor.astype.return_value = input_tensor

        # Mock randn to always return zero norm
        mock_u_tensor = Mock()
        mock_u_norm = Mock()
        mock_u_norm.asnumpy.return_value = 0.0
        mock_u_tensor.norm.return_value = mock_u_norm
        mock_randn.return_value = mock_u_tensor

        result = _get_max_eigenvalue(input_tensor, num_iter=2)
        assert result == 0.0
        mock_logger.warning.assert_called()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
