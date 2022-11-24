from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.PROCESSOR)
class BaseProcessor:
    def __init__(self):
        pass

