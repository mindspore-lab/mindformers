from mindspore import nn

from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.MODULES)
class MLP(nn.Cell):
    pass
