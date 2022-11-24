from mindspore import nn
from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.MODULES)
class Attention(nn.Cell):
    pass


@XFormerRegister.register(XFormerModuleType.MODULES)
class WindowAttention(nn.Cell):
    pass


