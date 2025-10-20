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
"""test weight loader in safetensors format"""
import os
import shutil
import tempfile
from safetensors import safe_open
from safetensors.numpy import save_file
import numpy as np
import pytest

from mindformers.parallel_core.inference.parallel_state import ProcessGroup
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.tensor_parallel.layers import (QKVParallelLinear, MergedColumnParallelLinear,
                                                                        RowParallelLinear)
from mindformers.parallel_core.inference.tensor_parallel.grouped_layers import (ColumnParallelGroupedLinear,
                                                                                RowParallelGroupedLinear)


class TestWeightLoader:
    """Base test class for weight loader tests."""

    def setup_method(self):
        # Mock process groups
        self.tp_group1 = ProcessGroup(size=2, rank=0)
        self.tp_group2 = ProcessGroup(size=2, rank=1)

        # config
        self.config = TransformerConfig(
            tensor_model_parallel_size=1,
            hidden_size=2,
            ffn_hidden_size=2,
            num_attention_heads=2,
            num_query_groups=2,
            kv_channels=1,
            num_moe_experts=2,
            moe_ffn_hidden_size=4,
            add_bias_linear=False,
            gated_linear_unit=True,
            hidden_act="silu",
            num_layers=1,
            compute_dtype='bf16',
            params_dtype='fp32',
        )


class TestQKVWeightLoader(TestWeightLoader):
    """Test weight loading for QKVParallelLinear."""

    def _create_qkv_layers(self):
        """
        Create two QKV parallel linear layer instances.

        This function initializes two QKVParallelLinear layers for different tensor parallel groups.
        Each layer creates query (Q), key (K), and value (V) parallel linear transformations
        based on the configuration parameters.

        Returns:
            tuple: A tuple containing two QKVParallelLinear instances (qkv1, qkv2)
                  qkv1: QKV layer using tp_group1 tensor parallel group
                  qkv2: QKV layer using tp_group2 tensor parallel group
        """
        qkv1 = QKVParallelLinear(
            hidden_size=self.config.hidden_size,
            head_size=self.config.kv_channels,
            total_num_heads=self.config.num_attention_heads,
            total_num_kv_heads=self.config.num_query_groups,
            config=self.config,
            tp_group=self.tp_group1
        )
        qkv2 = QKVParallelLinear(
            hidden_size=self.config.hidden_size,
            head_size=self.config.kv_channels,
            total_num_heads=self.config.num_attention_heads,
            total_num_kv_heads=self.config.num_query_groups,
            config=self.config,
            tp_group=self.tp_group2
        )
        return qkv1, qkv2

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        "shard_id, input_weights, expected_weights",
        [
            ('q',
             np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
             (
                 np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float16),
                 np.array([[3.0, 4.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float16)
             )),
            ('k',
             np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float16),
             (
                 np.array([[0.0, 0.0], [5.0, 6.0], [0.0, 0.0]], dtype=np.float16),
                 np.array([[0.0, 0.0], [7.0, 8.0], [0.0, 0.0]], dtype=np.float16)
             )),
            ('v',
             np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float16),
             (
                 np.array([[0.0, 0.0], [0.0, 0.0], [9.0, 10.0]], dtype=np.float16),
                 np.array([[0.0, 0.0], [0.0, 0.0], [11.0, 12.0]], dtype=np.float16)
             )),
        ]
    )
    def test_qkv_weight_loader(self, shard_id, input_weights, expected_weights):
        """
        Test QKV weight loading functionality to verify that weights are correctly loaded
        into the corresponding QKV layers under different shard_id scenarios.

        Args:
            shard_id (str): Identifier for the current weight type being processed.
                           Valid values are 'q', 'k', or 'v'.
            input_weights (np.ndarray): Input weight matrix with shape (2, 2) and data type float16.
            expected_weights (tuple of np.ndarray): Expected weight results for two QKV layers,
                                                   each element is an array with shape (3, 2).

        """
        qkv1, qkv2 = self._create_qkv_layers()
        qkv1.weight_loader(qkv1.weight, input_weights, shard_id)
        qkv2.weight_loader(qkv2.weight, input_weights, shard_id)

        expect_qkv1, expect_qkv2 = expected_weights
        assert np.array_equal(qkv1.weight.asnumpy(), expect_qkv1), f"QKV1 weight mismatch for shard_id='{shard_id}'"
        assert np.array_equal(qkv2.weight.asnumpy(), expect_qkv2), f"QKV2 weight mismatch for shard_id='{shard_id}'"


class TestMLPWeightLoader(TestWeightLoader):
    """Test weight loading for MergedColumnParallelLinear (MLP)."""

    def _create_mlp_layers(self):
        """
        Create two MLP layer instances.

        This function initializes two MergedColumnParallelLinear type MLP layers for processing
        the mapping from hidden layer to feed-forward network hidden layer.
        Each MLP layer uses the same configuration parameters but different tensor parallel groups.

        Returns:
            tuple: A tuple containing two MergedColumnParallelLinear instances (mlp1, mlp2)
        """
        mlp1 = MergedColumnParallelLinear(
            hidden_size=self.config.hidden_size,
            ffn_hidden_size=self.config.ffn_hidden_size,
            config=self.config,
            tp_group=self.tp_group1
        )
        mlp2 = MergedColumnParallelLinear(
            hidden_size=self.config.hidden_size,
            ffn_hidden_size=self.config.ffn_hidden_size,
            config=self.config,
            tp_group=self.tp_group2
        )
        return mlp1, mlp2

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        "shard_id, input_weights, expected_weights",
        [
            ('gating',
             np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
             (
                 np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float16),
                 np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float16)
             )),
            ('hidden',
             np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float16),
             (
                 np.array([[0.0, 0.0], [5.0, 6.0]], dtype=np.float16),
                 np.array([[0.0, 0.0], [7.0, 8.0]], dtype=np.float16)
             )),
        ]
    )
    def test_mlp_weight_loader(self, shard_id, input_weights, expected_weights):
        """
        Test MLP weight loading functionality to verify that weights are correctly loaded
        into the corresponding MLP layers under different shard_id scenarios.

        Args:
            shard_id (str): Identifier for the current weight type being processed.
                          Valid values are 'gating' or 'hidden'.
            input_weights (np.ndarray): Input weight matrix with shape (2, 2) and data type float16.
            expected_weights (tuple of np.ndarray): Expected weight results for two MLP layers,
            each element is an array with shape (2, 2).

        Functionality:
            - Creates two MLP layer instances.
            - Uses the weight_loader method to load input_weights to the corresponding shard_id position.
            - Verifies that the loaded weights match the expected results.
        """
        mlp1, mlp2 = self._create_mlp_layers()
        mlp1.weight_loader(mlp1.weight, input_weights, shard_id)
        mlp2.weight_loader(mlp2.weight, input_weights, shard_id)

        expect_mlp1, expect_mlp2 = expected_weights
        assert np.array_equal(mlp1.weight.asnumpy(), expect_mlp1), f"MLP1 weight mismatch for shard_id='{shard_id}'"
        assert np.array_equal(mlp2.weight.asnumpy(), expect_mlp2), f"MLP2 weight mismatch for shard_id='{shard_id}'"


class TestMOEColumnWeightLoader(TestWeightLoader):
    """Test weight loading for ColumnParallelGroupedLinear (MOE)."""

    def setup_method(self):
        super().setup_method()

        self._weight_cache = {}

        # MOE data
        self.test_data = {
            "w1_expert_0": np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
            "w1_expert_1": np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
            "w3_expert_0": np.array([[5.0, 6.0], [7.0, 8.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float16),
            "w3_expert_1": np.array([[5.0, 6.0], [7.0, 8.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float16),
        }

        self.temp_work_dir = tempfile.mkdtemp()
        self.moe_safetensors_path = os.path.join(self.temp_work_dir, "test_moe_weights.safetensors")
        save_file(self.test_data, self.moe_safetensors_path)


    def load_test_weights(self):
        if not self._weight_cache:
            with safe_open(self.moe_safetensors_path, framework="np") as f:
                for key in f.keys():
                    self._weight_cache[key] = f.get_slice(key)
        return self._weight_cache

    def _create_moe_layers(self):
        """
        Create two MoE (Mixture of Experts) layers.

        This function initializes two column parallel grouped linear layers for implementing
        the expert networks in the MoE architecture. Each layer is configured with the same
        number of experts, input/output dimensions, and other parameters.

        Returns:
            tuple: A tuple containing two ColumnParallelGroupedLinear instances
                - moe1: First MoE layer using tp_group1 as tensor parallel group
                - moe2: Second MoE layer using tp_group2 as tensor parallel group
        """
        moe1 = ColumnParallelGroupedLinear(
            num_local_experts=self.config.num_moe_experts,
            input_size=self.config.hidden_size,
            output_size=2 * self.config.moe_ffn_hidden_size,
            config=self.config,
            tp_group=self.tp_group1
        )
        moe2 = ColumnParallelGroupedLinear(
            num_local_experts=self.config.num_moe_experts,
            input_size=self.config.hidden_size,
            output_size=2 * self.config.moe_ffn_hidden_size,
            config=self.config,
            tp_group=self.tp_group2
        )
        return moe1, moe2

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        "shard_id, weight_key, expected_weights, expert_id",
        [
            ('w1', 'w1_expert_0',
             (
                 np.array([[[1.0, 3.0, 0.0, 0.0], [2.0, 4.0, 0.0, 0.0]],
                           [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=np.float16),
                 np.array([[[1.0, 3.0, 0.0, 0.0], [2.0, 4.0, 0.0, 0.0]],
                           [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=np.float16),
             ),
             0),
            ('w1', 'w1_expert_1',
             (
                 np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                           [[1.0, 3.0, 0.0, 0.0], [2.0, 4.0, 0.0, 0.0]]], dtype=np.float16),
                 np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                           [[1.0, 3.0, 0.0, 0.0], [2.0, 4.0, 0.0, 0.0]]], dtype=np.float16),
             ),
             1),
            ('w3', 'w3_expert_0',
             (
                 np.array([[[0.0, 0.0, 5.0, 7.0], [0.0, 0.0, 6.0, 8.0]],
                           [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=np.float16),
                 np.array([[[0.0, 0.0, 5.0, 7.0], [0.0, 0.0, 6.0, 8.0]],
                           [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=np.float16),
             ),
             0),
            ('w3', 'w3_expert_1',
             (
                 np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                           [[0.0, 0.0, 5.0, 7.0], [0.0, 0.0, 6.0, 8.0]]], dtype=np.float16),
                 np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                           [[0.0, 0.0, 5.0, 7.0], [0.0, 0.0, 6.0, 8.0]]], dtype=np.float16),
             ),
             1),
        ]
    )
    def test_moe_weight_loader(self, shard_id, weight_key, expected_weights, expert_id):
        """
        Test MOE weight loading functionality to verify that weights are correctly loaded
        into the corresponding MOE layers for different shard_id and expert_id scenarios.

        Args:
            shard_id (str): Identifier for the current weight type being processed.
                           Valid values are 'w1' or 'w3' for column parallel layers.
            weight_key (str): Key to retrieve the input weights from the test data cache.
            expected_weights (tuple of np.ndarray): Expected weight results for two MOE layers,
                                                   each element is a 3D array with shape (2, 2, 4).
            expert_id (int): Identifier for the expert being processed (0 or 1).
        """
        input_weights = self.load_test_weights()[weight_key]

        moe1, moe2 = self._create_moe_layers()
        moe1.weight_loader(moe1.weight, input_weights, shard_id, expert_id)
        moe2.weight_loader(moe2.weight, input_weights, shard_id, expert_id)

        expect_moe1, expect_moe2 = expected_weights
        assert np.array_equal(moe1.weight.asnumpy(), expect_moe1), f"MOE1 weight mismatch for shard_id='{shard_id}'"
        assert np.array_equal(moe2.weight.asnumpy(), expect_moe2), f"MOE2 weight mismatch for shard_id='{shard_id}'"

    def teardown_method(self):
        """remove fake safetensors directory"""
        shutil.rmtree(self.temp_work_dir)


class TestRowLinearWeightLoader(TestWeightLoader):
    """Test weight loading for RowParallelLinear."""

    def _create_row_linear_layers(self):
        """
        Create two row parallel linear layer instances.

        This function initializes two RowParallelLinear layers for different tensor parallel groups.
        Each layer performs row parallel linear transformation with the same input and output dimensions.

        Returns:
            tuple: A tuple containing two RowParallelLinear instances (row1, row2)
                  row1: Row parallel linear layer using tp_group1 tensor parallel group
                  row2: Row parallel linear layer using tp_group2 tensor parallel group
        """
        row1 = RowParallelLinear(
            input_size=self.config.hidden_size,
            output_size=self.config.hidden_size,
            config=self.config,
            tp_group=self.tp_group1
        )
        row2 = RowParallelLinear(
            input_size=self.config.hidden_size,
            output_size=self.config.hidden_size,
            config=self.config,
            tp_group=self.tp_group2
        )
        return row1, row2

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        "input_weights, expected_weights",
        [
            (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
             (
                 np.array([[1.0], [3.0]], dtype=np.float16),
                 np.array([[2.0], [4.0]], dtype=np.float16)
             )),
        ]
    )
    def test_row_linear_weight_loader(self, input_weights, expected_weights):
        """
        Test row linear weight loading functionality to verify that weights are correctly loaded
        into the corresponding row linear layers.

        Args:
            input_weights (np.ndarray): Input weight matrix with shape (2, 2) and data type float16.
            expected_weights (tuple of np.ndarray): Expected weight results for two row linear layers,
                                                   each element is an array with shape (2, 1).

        Functionality:
            - Creates two row linear layer instances.
            - Uses the weight_loader method to load input_weights into both layers.
            - Verifies that the loaded weights match the expected results.
        """
        row1, row2 = self._create_row_linear_layers()
        row1.weight_loader(row1.weight, input_weights)
        row2.weight_loader(row2.weight, input_weights)

        expect_row1, expect_row2 = expected_weights
        assert np.array_equal(row1.weight.asnumpy(), expect_row1), f"ROW1 weight mismatch'"
        assert np.array_equal(row2.weight.asnumpy(), expect_row2), f"ROW2 weight mismatch'"


class TestMOERowWeightLoader(TestWeightLoader):
    """Test weight loading for RowParallelGroupedLinear (MOE)."""

    def _create_moe_down_layers(self):
        """
        Create MoE down-sampling layers.

        This method creates two row parallel grouped linear layers for down-sampling operations
        in the MoE architecture. Each layer is configured with the same expert count, input/output
        dimensions and other parameters.

        Returns:
            tuple: A tuple containing two RowParallelGroupedLinear layers (down1, down2)
                - down1: First row parallel grouped linear layer using tp_group1
                - down2: Second row parallel grouped linear layer using tp_group2
        """
        down1 = RowParallelGroupedLinear(
            num_local_experts=self.config.num_moe_experts,
            input_size=self.config.hidden_size,
            output_size=self.config.moe_ffn_hidden_size,
            config=self.config,
            tp_group=self.tp_group1
        )
        down2 = RowParallelGroupedLinear(
            num_local_experts=self.config.num_moe_experts,
            input_size=self.config.hidden_size,
            output_size=self.config.moe_ffn_hidden_size,
            config=self.config,
            tp_group=self.tp_group2
        )
        return down1, down2

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        "shard_id, input_weights, expected_weights, expert_id",
        [
            ('w2',
             np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
             (
                 np.array([[[1.0, 3.0, 1.0, 3.0]], [[0.0, 0.0, 0.0, 0.0]]], dtype=np.float16),
                 np.array([[[2.0, 4.0, 2.0, 4.0]], [[0.0, 0.0, 0.0, 0.0]]], dtype=np.float16),
             ),
             0),
            ('w2',
             np.array([[5.0, 6.0], [7.0, 8.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float16),
             (
                 np.array([[[0.0, 0.0, 0.0, 0.0]], [[5.0, 7.0, 5.0, 7.0]]], dtype=np.float16),
                 np.array([[[0.0, 0.0, 0.0, 0.0]], [[6.0, 8.0, 6.0, 8.0]]], dtype=np.float16),
             ),
             1),
        ]
    )
    def test_moe_down_weight_loader(self, shard_id, input_weights, expected_weights, expert_id):
        """
        Test MOE down weight loading functionality to verify that weights are correctly loaded
        into the corresponding MOE down layers for different shard_id and expert_id scenarios.

        Args:
            shard_id (str): Identifier for the current weight type being processed.
                           Valid value is 'w2' for down layers.
            input_weights (np.ndarray): Input weight matrix with shape (2, 2) and data type float16.
            expected_weights (tuple of np.ndarray): Expected weight results for two MOE down layers,
                                                   each element is a 3D array.
            expert_id (int): Identifier for the expert being processed (0 or 1).

        Functionality:
            - Creates two MOE down layer instances.
            - Uses the weight_loader method to load input_weights to the corresponding
              shard_id and expert_id positions.
            - Verifies that the loaded weights match the expected results.
        """
        down1, down2 = self._create_moe_down_layers()
        down1.weight_loader(down1.weight, input_weights, shard_id, expert_id)
        down2.weight_loader(down2.weight, input_weights, shard_id, expert_id)

        expect_down1, expect_down2 = expected_weights
        assert np.array_equal(down1.weight.asnumpy(),
                              expect_down1), f"MOE_DOWN1 weight mismatch for shard_id='{shard_id}'"
        assert np.array_equal(down2.weight.asnumpy(),
                              expect_down2), f"MOE_DOWN2 weight mismatch for shard_id='{shard_id}'"
