"""MindFormers Moe API."""
from mindspore import nn

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.MODULES)
class Moe(nn.Cell):
    """MOE."""
