"""Build LR Schedule API."""
import inspect

from mindspore import nn

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_lr(
        config: dict = None, default_args: dict = None,
        module_type: str = 'lr', class_name: str = None, **kwargs):
    """Build LR API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.LR, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_lr():
    """ register MindSpore builtin LR class. """
    for module_name in dir(nn.learning_rate_schedule):
        if module_name.startswith('__'):
            continue
        lr_schedule = getattr(nn.learning_rate_schedule, module_name)
        if inspect.isclass(lr_schedule):
            MindFormerRegister.register_cls(
                lr_schedule, MindFormerModuleType.LR)


register_ms_lr()
