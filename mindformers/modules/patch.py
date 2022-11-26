"""MindFormers Patch API."""
from mindspore import nn

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.MODULES)
class PatchEmbed(nn.Cell):
    """Patch Embed."""
