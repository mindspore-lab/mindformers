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
from mindspore.ops import operations as P

from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.WRAPPER)
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
