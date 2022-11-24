from mindspore import nn

from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.MODULES)
class Moe(nn.Cell):
    pass
