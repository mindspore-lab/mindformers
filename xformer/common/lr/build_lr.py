import inspect

from mindspore import nn

from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_lr(config: dict = None, default_args: dict = None,
             module_type: str = 'lr', class_name: str = None,
             *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        return XFormerRegister.get_instance_from_cfg(
            config, XFormerModuleType.LR, default_args=default_args)
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def register_ms_lr():
    """ register MindSpore builtin LR class. """
    for module_name in dir(nn.learning_rate_schedule):
        if module_name.startswith('__'):
            continue
        lr = getattr(nn.learning_rate_schedule, module_name)
        if inspect.isclass(lr):
            XFormerRegister.register_cls(
                lr, XFormerModuleType.LR)


register_ms_lr()
