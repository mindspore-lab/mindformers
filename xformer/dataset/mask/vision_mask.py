from mindspore.dataset.transforms import py_transforms

from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.MASK_POLICY)
class SimMask(py_transforms.PyTensorOperation):
    pass


@XFormerRegister.register(XFormerModuleType.MASK_POLICY)
class MaeMask(py_transforms.PyTensorOperation):
    pass

