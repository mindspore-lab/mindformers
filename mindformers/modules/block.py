"""MindFormers Block API."""
from mindspore import nn

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.MODULES)
class Block(nn.Cell):
    """Block."""
