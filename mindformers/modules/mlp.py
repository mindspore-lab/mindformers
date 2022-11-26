"""MindFormers MLP API."""
from mindspore import nn

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.MODULES)
class MLP(nn.Cell):
    """MLP."""
