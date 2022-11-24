import inspect

from mindspore import nn

from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_layer(config: dict = None, default_args: dict = None,
                module_type: str = 'base_layer', class_name: str = None,
                *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        return XFormerRegister.get_instance_from_cfg(
            config, XFormerModuleType.BASE_LAYER, default_args=default_args)
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def register_ms_layer():
    """ register MindSpore builtin base layer class. """
    for module_name in dir(nn.layer):
        if module_name.startswith('__'):
            continue
        ly = getattr(nn.layer, module_name)
        if inspect.isclass(ly):
            XFormerRegister.register_cls(
                ly, XFormerModuleType.BASE_LAYER)


register_ms_layer()
