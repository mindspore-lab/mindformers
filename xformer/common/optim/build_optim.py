import inspect

from mindspore import nn

from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_optim(config: dict = None, default_args: dict = None,
                module_type: str = 'optimizer', class_name: str = None,
                *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        return XFormerRegister.get_instance_from_cfg(
            config, XFormerModuleType.OPTIMIZER, default_args=default_args)
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def register_ms_optim():
    """ register MindSpore builtin optimizer class. """
    for module_name in dir(nn.optim):
        if module_name.startswith('__'):
            continue
        optim = getattr(nn.optim, module_name)
        if inspect.isclass(optim):
            XFormerRegister.register_cls(
                optim, XFormerModuleType.OPTIMIZER)


register_ms_optim()
