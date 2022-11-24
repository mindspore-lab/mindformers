from typing import Callable, List

from xformer.trainer.base_trainer import BaseTrainer
from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.TRAINER, alias="mim")
class MaskedImageModelingTrainer(BaseTrainer):
    def __init__(self, model_name: str = None):
        super(MaskedImageModelingTrainer, self).__init__(model_name)
        self.model_name = model_name

    def train(self,
              config: dict = None,
              network: Callable = None,
              dataset: Callable = None,
              optimizer: Callable = None,
              callbacks: List[Callable] = None, **kwargs):
        # 自定义创建模型训练完整过程, 待补充
        pass

    def evaluate(self,
                 config: dict = None,
                 network: Callable = None,
                 dataset: Callable = None,
                 callbacks: List[Callable] = None, **kwargs):
        # 自定义创建模型评估完整过程, 待补充
        pass
