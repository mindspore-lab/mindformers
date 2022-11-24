from .mae_config import MaeConfig
from xformer.models.base_model import BaseModel

from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.ENCODER)
class MaeEncoder(BaseModel):
    """vision xformer for mae"""

    def __init__(self, config=MaeConfig()):
        super().__init__()
        self.config = config
