# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Callback Function"""
import time
from mindspore.train.summary import SummaryRecord
from mindspore.train.callback import Callback


def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note:
        If per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_id=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_id
        self.time_stamp_first = get_ms_timestamp()

    def step_end(self, run_context):
        """Monitor the loss in training."""
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - self.time_stamp_first,
                                                                     cb_params.cur_epoch_num,
                                                                     cb_params.cur_step_num,
                                                                     str(cb_params.net_outputs)))

        loss_file = "./loss_{}.log"

        with open(loss_file.format(self.rank_id), "a+") as f:
            f.write("time: {} ms, epoch: {}, step: {}, loss: {}, overflow: {}, loss_scale: {}".format(
                time_stamp_current - self.time_stamp_first,
                cb_params.cur_epoch_num,
                cb_params.cur_step_num,
                str(cb_params.net_outputs[0].asnumpy()),
                str(cb_params.net_outputs[1].asnumpy()),
                str(cb_params.net_outputs[2].asnumpy())))
            f.write('\n')


class LossSummaryCallback(Callback):
    """
    A basic summary writer recording the loss

    Args:
        summary_dir (str) : The path to store the summary dir

    """
    def __init__(self, summary_dir):
        self._summary_dir = summary_dir

    def __enter__(self):
        """
        Init the summary record in here, when the train script run, it will be inited before training
        """
        self.summary_record = SummaryRecord(self._summary_dir)
        return self

    def __exit__(self, *exc_args):
        """
        Note: must close the summary record, it will release the process pool resource
        else the training script will not exit from training.
        """
        self.summary_record.close()

    def step_end(self, run_context):
        """
        Print information at the end of step
        """
        cb_params = run_context.original_args()
        outputs = cb_params.net_outputs
        if isinstance(outputs, (tuple, list)):
            loss = outputs[0]
        else:
            loss = outputs
        self.summary_record.add_value('scalar', 'loss', loss)
        self.summary_record.record(cb_params.cur_step_num)
        self.summary_record.flush()
