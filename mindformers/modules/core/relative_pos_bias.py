"""RelativePosition Core API."""
from mindspore import nn

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.CORE)
class RelativePositionBias(nn.Cell):
    """RelativePositionBias."""
