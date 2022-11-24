from mindspore import nn

from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.CORE)
class RelativePositionBias(nn.Cell):
    pass
