
from mindspore import nn

from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.MODULES)
class PatchEmbed(nn.Cell):
    pass


@XFormerRegister.register(XFormerModuleType.MODULES)
class Patchify(nn.Cell):
    pass


@XFormerRegister.register(XFormerModuleType.MODULES)
class UnPatchify(nn.Cell):
    pass
