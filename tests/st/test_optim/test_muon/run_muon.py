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
"""Run Muon optimizer accuracy test with configurable parameters via args"""
import argparse
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

from mindformers.core.context.build_context import build_context
from mindformers.core.optim.muon import Muon

np.random.seed(1024)

# Test weight initialization - same as optimizer_util.py
FC1_WEIGHT = np.array([[0.72346634, 0.95608497, 0.4084163, 0.18627149,
                        0.6942514, 0.39767185, 0.24918061, 0.4548748],
                       [0.7203382, 0.19086994, 0.76286614, 0.87920564,
                        0.3169892, 0.9462494, 0.62827677, 0.27504718],
                       [0.3544535, 0.2524781, 0.5370583, 0.8313121,
                        0.6670143, 0.0488653, 0.62225235, 0.7546456],
                       [0.17985944, 0.05106374, 0.31064633, 0.4863033,
                        0.848814, 0.5523157, 0.20295663, 0.7213356]]).astype("float32")

FC1_BIAS = np.array([0.79708564, 0.13728078, 0.66322654, 0.88128525]).astype("float32")

FC2_WEIGHT = np.array([[0.8473515, 0.50923985, 0.42287776, 0.29769543]]).astype("float32")

FC2_BIAS = np.array([0.09996348]).astype("float32")


class MockTransformerConfig:
    """Mock transformer config for testing Muon optimizer."""
    def __init__(self):
        self.multi_latent_attention = True
        self.tensor_model_parallel_size = 1
        self.data_parallel_size = 1


class MockModel:
    """
    Mock model class that provides required interfaces for Muon optimizer.
    This simulates the model interface that Muon optimizer expects.
    """
    def __init__(self):
        self.config = MockTransformerConfig()

    def get_gpt_transformer_config(self):
        """Return transformer config."""
        return self.config

    def make_model_muon_fns(self):
        """Return muon split and merge functions."""
        def muon_split_fn(param_name, tensor):  # pylint: disable=unused-argument
            """Split function - returns tensor as list."""
            return [tensor]

        def muon_merge_fn(param_name, tensor_list):  # pylint: disable=unused-argument
            """Merge function - returns first tensor."""
            return tensor_list[0]

        return muon_split_fn, muon_merge_fn

    # pylint: disable=unused-argument
    def apply_qk_clip_scaling(self, params, param_names, param_layer, logit_threshold,
                               muon_split_fn, muon_merge_fn):
        """Apply query-key clipping scaling."""
        return [(0, params[0])]

    def get_param_layer_indices(self, params):
        """Return layer indices for parameters."""
        return {p.name: 0 for p in params}

    def get_muon_filter(self):
        """Return filter function to determine which params use Muon."""
        def muon_filter(param):
            # Apply Muon to weight parameters with 2D shape (not bias)
            return len(param.shape) == 2 and 'bias' not in param.name
        return muon_filter

    def get_tp_dims(self, params):
        """Return tensor parallel dimensions."""
        return tuple(-1 for _ in params)

    def get_op_groups_info(self, params, op):  # pylint: disable=unused-argument
        """Return optimizer parallel group info."""
        ops = tuple(1 for _ in params)
        op_groups = tuple("" for _ in params)
        return ops, op_groups


class FakeNet(nn.Cell):
    """Build fake net for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(in_channels=8, out_channels=4,
                            weight_init=Tensor(FC1_WEIGHT),
                            bias_init=Tensor(FC1_BIAS))
        self.fc2 = nn.Dense(in_channels=4, out_channels=1,
                            weight_init=Tensor(FC2_WEIGHT),
                            bias_init=Tensor(FC2_BIAS))
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetWithLoss(nn.Cell):
    """Build net with loss."""

    def __init__(self, network, loss_fn):
        super().__init__()
        self.network = network
        self.loss = loss_fn

    def construct(self, x, label):
        out = self.network(x)
        loss = self.loss(out, label)
        return loss


def make_fake_data():
    """Make fake data for testing."""
    data, label = [], []
    for i in range(20):
        data.append(ms.Tensor(np.array(np.ones((2, 8)) * i, dtype=np.float32)))
        label.append(ms.Tensor(np.array(np.ones((2, 1)) * (i + 1), dtype=np.float32)))
    return data, label


class MuonRunner:
    """Class to manage Muon optimizer test and training."""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.learning_rate = self.args.learning_rate
        self.weight_decay = self.args.weight_decay
        self.momentum = self.args.momentum
        self.nesterov = self.args.nesterov
        self.num_steps = self.args.num_steps

    def build_network(self):
        """Build network with Muon optimizer."""
        net = FakeNet()
        mock_model = MockModel()

        loss_fn = nn.L1Loss(reduction='mean')
        networkwithloss = NetWithLoss(net, loss_fn)
        networkwithloss.set_train()

        params = networkwithloss.trainable_params()

        # Create Muon optimizer
        optimizer = Muon(
            params=params,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            matched_adamw_rms=0.2,
            momentum=self.momentum,
            nesterov=self.nesterov,
            adamw_betas=(0.95, 0.95),
            adamw_eps=1e-8,
            model=mock_model,
        )

        return networkwithloss, optimizer, mock_model

    def run(self):
        """Run the training with Muon optimizer."""
        networkwithloss, optimizer, mock_model = self.build_network()
        trainonestepcell = nn.TrainOneStepCell(networkwithloss, optimizer)

        losses = []
        data, label = make_fake_data()
        for i in range(self.num_steps):
            loss = trainonestepcell(data[i], label[i])
            losses.append(loss.asnumpy())

        # Save results
        output_dict = {
            "losses": np.array(losses),
            "num_muon_m": len(optimizer.muon_m),
            "num_moments1": len(optimizer.moments1),
            "num_moments2": len(optimizer.moments2),
        }

        # Save muon momentum values for weight parameters
        muon_filter = mock_model.get_muon_filter()
        # pylint: disable=protected-access
        for idx, param in enumerate(optimizer._parameters):
            if muon_filter(param):
                muon_m_value = optimizer.muon_m[idx].asnumpy()
                output_dict[f"muon_m_{idx}"] = muon_m_value

        np.savez(self.args.output_path, **output_dict)
        print(f"Results saved to {self.args.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Muon optimizer test")
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--nesterov", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--output_path", type=str, default="output_muon.npz")

    args = parser.parse_args()

    # Set context
    build_context({"use_legacy": False, "use_parallel": True})
    ms.set_deterministic(True)
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_seed(42)

    # Run training
    runner = MuonRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
