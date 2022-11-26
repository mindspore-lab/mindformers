"""Self-Define Vision Mask Policy."""
from mindspore.dataset.transforms import py_transforms
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.MASK_POLICY)
class SimMask(py_transforms.PyTensorOperation):
    """SimMIM Mask Policy."""


@MindFormerRegister.register(MindFormerModuleType.MASK_POLICY)
class MaeMask(py_transforms.PyTensorOperation):
    """MAE Mask Policy."""
