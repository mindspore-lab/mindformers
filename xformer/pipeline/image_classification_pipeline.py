from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.PIPELINE)
class ImageClassificationForPipeline:
    def __init__(self):
        pass
