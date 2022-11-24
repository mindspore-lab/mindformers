from mindspore import nn

from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.MODULES)
class SwinTransformerBlock(nn.Cell):
    pass


@XFormerRegister.register(XFormerModuleType.MODULES)
class VisionTransformerBlock(nn.Cell):
    pass
