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
Baseline data for Muon optimizer tests.
"""
import numpy as np

# Default tolerance for loss comparison
DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-4

# Baseline losses for single card test cases
# learning_rate=0.02, weight_decay=0.1, momentum=0.95, nesterov=True
BASELINE_LOSSES_NESTEROV_TRUE = np.array([
    0.3881023, 7.8122883, 15.039654, 22.062939, 28.884716,
    35.514862, 41.940598, 48.178577, 54.222153, 60.07846,
    65.739815, 71.20518, 76.508705, 81.63688, 86.58084,
    91.356064, 95.94581, 100.37069, 104.620384, 108.72005
], dtype=np.float32)

# learning_rate=0.02, weight_decay=0.1, momentum=0.95, nesterov=False
BASELINE_LOSSES_NESTEROV_FALSE = np.array([
    0.3881023, 7.8122883, 15.032751, 22.052126, 28.875042,
    35.503002, 41.92948, 48.16231, 54.218227, 60.07244,
    65.745224, 71.22119, 76.5374, 81.64788, 86.525246,
    91.292816, 95.89634, 100.308716, 104.57111, 108.64668
], dtype=np.float32)

# learning_rate=0.01, weight_decay=0.05, momentum=0.9, nesterov=True
BASELINE_LOSSES_DIFF_LR = np.array([
    0.3881023, 7.8966713, 15.322964, 22.66404, 29.917278,
    37.085056, 44.168663, 51.175865, 58.094597, 64.92998,
    71.680595, 78.34835, 84.92714, 91.44285, 97.866035,
    104.204056, 110.46475, 116.63603, 122.729706, 128.74644
], dtype=np.float32)


def compare_losses(actual_losses, expected_losses, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """
    Compare actual losses with expected baseline losses.

    Args:
        actual_losses (np.ndarray): Actual losses from the test run
        expected_losses (np.ndarray): Expected baseline losses
        rtol (float): Relative tolerance for comparison
        atol (float): Absolute tolerance for comparison

    Returns:
        bool: True if losses match within tolerance, False otherwise
    """
    return np.allclose(actual_losses, expected_losses, rtol=rtol, atol=atol)
