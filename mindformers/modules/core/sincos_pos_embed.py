"""SinCosPE2D Core API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.CORE)
class SinCosPE2D:
    """SinCosPE2D."""
