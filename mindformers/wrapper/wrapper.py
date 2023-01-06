# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Self-Define Wrapper."""
from mindspore import nn
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P


from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


__all__ = ['ClassificationMoeWrapper', 'TrainOneStepWithClipGN']


@MindFormerRegister.register(MindFormerModuleType.WRAPPER)
class ClassificationMoeWrapper(nn.WithLossCell):
    """Image Classification With Moe Module."""
    def __init__(self, backbone, loss_fn):
        super(ClassificationMoeWrapper, self).__init__(backbone, loss_fn)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self._add = P.Add().shard(((), ()))

    def construct(self, data, label):
        out, moe_loss = self._backbone(data)
        loss = self._loss_fn(out, label)
        return self._add(loss, moe_loss)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, the backbone network.
        """
        return self._backbone


_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)


@MindFormerRegister.register(MindFormerModuleType.WRAPPER)
class TrainOneStepWithClipGN(nn.TrainOneStepWithLossScaleCell):
    """TrainOneStep"""

    def __init__(self, network, optimizer,
                 use_clip_grad=False, clip_norm=1.0,
                 scale_sense=1.0):
        super(TrainOneStepWithClipGN, self).__init__(network, optimizer, scale_sense)
        self.print = P.Print()
        self.clip_norm = clip_norm
        self.use_clip_grad = use_clip_grad

    def construct(self, *inputs):
        """construct"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            if self.use_clip_grad:
                grads = C.clip_by_global_norm(grads, clip_norm=self.clip_norm)
            loss = F.depend(loss, self.optimizer(grads))
        else:
            self.print("==========Overflow Now============")
        return loss
