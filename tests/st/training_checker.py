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
"""Training checker"""
import time

from mindspore import Callback

from mindformers.core.callback.callback import _get_loss_output


class TrainingChecker(Callback):
    """
    Callback function for precision and performance checking. Raise an AssertionError once the difference
    between a step's loss and the corresponding expected value is greater than the error value or the
    difference ratio between average step time and expected value is greater than the error ratio.

    Args:
        loss_list_std (list[float]):
            A list of expected loss values.
        avg_step_time_std (float):
            expected average step time value (in millisecond).
        loss_error (float, optional):
            Allowable loss error between true and expected values. Defaults to 1e-3.
        time_error_ratio (float, optional):
            Allowable time error ratio between true and expected values. Defaults to 0.1.

    Raises:
        AssertionError
    """
    def __init__(self, loss_list_std: list, avg_step_time_std: float,
                 loss_error: float = 1e-3, time_error_ratio: float = 0.1):
        super(TrainingChecker, self).__init__()
        self.loss_list_std = loss_list_std
        self.avg_step_time_std = avg_step_time_std
        self.loss_error = loss_error
        self.time_error_ratio = time_error_ratio
        self.step_time = time.time()

    def on_train_step_begin(self, run_context):
        """Called on each training step begin."""
        _ = run_context
        self.step_time = time.time()

    def on_train_step_end(self, run_context):
        """Called on each training step end."""
        cb_params = run_context.original_args()
        net_outputs = cb_params.net_outputs
        loss = _get_loss_output(net_outputs)[0]
        cur_step_num = cb_params.cur_step_num
        cur_step_time = (time.time() - self.step_time) * 1000

        # when enable pp, loss will be only available on the last card
        if cb_params.parallel_mode != "stand_alone" and loss != 0.0:
            assert abs(loss - self.loss_list_std[cur_step_num - 1]) < self.loss_error

        if cur_step_num > 2:
            assert abs(cur_step_time - self.avg_step_time_std) / self.avg_step_time_std < self.time_error_ratio
