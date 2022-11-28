"""Base Processor API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class BaseProcessor:
    """Base Processor."""
    def __init__(self):
        pass
