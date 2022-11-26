"""Image Classification Pipeline API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.PIPELINE)
class ImageClassificationForPipeline:
    """Image Classification For Pipeline."""
    def __init__(self):
        pass
