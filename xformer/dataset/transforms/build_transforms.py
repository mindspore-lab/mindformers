import inspect

from mindspore.dataset import transforms as tf
from mindspore.dataset import vision as vs

from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_transforms(config: dict = None, default_args: dict = None,
                     module_type: str = 'transforms', class_name: str = None,
                     *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        cfg_transforms = config
        if not isinstance(cfg_transforms, list):
            return XFormerRegister.get_instance_from_cfg(
                cfg_transforms, XFormerModuleType.TRANSFORMS, default_args=default_args)
        transforms = []
        for transform in cfg_transforms:
            transform_op = XFormerRegister.get_instance_from_cfg(
                transform, XFormerModuleType.TRANSFORMS)
            transforms.append(transform_op)
        return transforms
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def register_ms_c_transforms():
    """register MindSpore builtin c_transforms class."""
    for module_name in set(dir(tf.c_transforms) + dir(vs.c_transforms)):
        if module_name.startswith('__'):
            continue

        c_transforms = getattr(tf.c_transforms, module_name, None) \
            if getattr(tf.c_transforms, module_name, None) else getattr(vs.c_transforms, module_name)
        if inspect.isclass(c_transforms):
            class_name = 'C_' + c_transforms.__name__
            XFormerRegister.register_cls(c_transforms, XFormerModuleType.TRANSFORMS, alias=class_name)


def register_ms_py_transforms():
    """register MindSpore builtin py_transforms class."""
    for module_name in set(dir(tf.py_transforms) + dir(vs.py_transforms)):
        if module_name.startswith('__'):
            continue

        py_transforms = getattr(tf.py_transforms, module_name, None) \
            if getattr(tf.py_transforms, module_name, None) else getattr(vs.py_transforms, module_name)
        if inspect.isclass(py_transforms):
            class_name = 'PY_' + py_transforms.__name__
            XFormerRegister.register_cls(py_transforms, XFormerModuleType.TRANSFORMS, alias=class_name)


register_ms_c_transforms()
register_ms_py_transforms()
