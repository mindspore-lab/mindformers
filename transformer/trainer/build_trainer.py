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
"""Used for building training process."""
from mindspore import context
import mindspore.common.dtype as mstype

from mindspore.nn.wrap.loss_scale import TrainOneStepCell

from .grad_accu_trainer import TrainAccuStepsWithLossScaleCell
from .trainer import TrainOneStepGradWithLossScaleCell


def build_trainer(config, net, optim, update_cell):
    """Build the TrainOneStepCell according to the """
    if context.get_context("device_target") == "CPU":
        config.logger.info("For training on cpu, the loss scale will always be 1.")
        step_cell = TrainOneStepCell(net, optim)
        return step_cell

    if config.acc_step > 1:
        step_cell = TrainAccuStepsWithLossScaleCell(net, optim, update_cell)
    else:
        step_cell = TrainOneStepGradWithLossScaleCell(net, optim, update_cell)

    if config.parallel_mode == context.ParallelMode.DATA_PARALLEL:
        step_cell.set_custom_sync_dtype(mstype.float16 if config.grad_sync_dtype == "fp16" else mstype.float32)
    return step_cell
