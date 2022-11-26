"""Image Classification Trainer."""
from typing import Callable, List

from mindformers.trainer.base_trainer import BaseTrainer
from mindformers.pipeline import ImageClassificationForPipeline
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.TRAINER, alias="image_classification")
class ImageClassificationTrainer(BaseTrainer):
    """Image Classification Trainer."""
    def __init__(self, model_name: str = None):
        super(ImageClassificationTrainer, self).__init__(model_name)
        self.model_name = model_name

    def train(self,
              config: dict = None,
              network: Callable = None,
              dataset: Callable = None,
              optimizer: Callable = None,
              callbacks: List[Callable] = None, **kwargs):
        """train for trainer."""
        # 自定义创建模型训练完整过程, 待补充

    def evaluate(self,
                 config: dict = None,
                 network: Callable = None,
                 dataset: Callable = None,
                 callbacks: List[Callable] = None, **kwargs):
        """evaluate for trainer."""
        # 自定义创建模型评估完整过程, 待补充

    def predict(self,
                config: dict = None,
                network: Callable = None,
                dataset: Callable = None, **kwargs):
        """predict for trainer."""

    # 直接使用pipeline流程进行定义, 待补充
    pipeline = ImageClassificationForPipeline()
