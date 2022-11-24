import inspect

from mindspore import nn

from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_loss(config: dict = None, default_args: dict = None,
               module_type: str = 'loss', class_name: str = None,
               *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        return XFormerRegister.get_instance_from_cfg(
            config, XFormerModuleType.LOSS, default_args=default_args)
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def register_ms_loss():
    """ register MindSpore builtin loss class. """
    for module_name in dir(nn.loss):
        if module_name.startswith('__'):
            continue
        loss = getattr(nn.loss, module_name)
        if inspect.isclass(loss):
            XFormerRegister.register_cls(
                loss, XFormerModuleType.LOSS)


register_ms_loss()
