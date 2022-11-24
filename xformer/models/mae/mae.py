
from xformer.models.base_model import BaseModel
from .mae_config import MaeConfig

from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.MODELS)
class Mae(BaseModel):
    """Pretrain MAE Module."""

    def __init__(self, config=MaeConfig()):
        super(Mae, self).__init__()
        self.config = config
