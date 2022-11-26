"""Build Layer API."""
import inspect

from mindspore import nn

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_layer(
        config: dict = None, default_args: dict = None,
        module_type: str = 'base_layer', class_name: str = None, **kwargs):
    """Build layer API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.BASE_LAYER, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_layer():
    """ register MindSpore builtin base layer class. """
    for module_name in dir(nn.layer):
        if module_name.startswith('__'):
            continue
        ms_layer = getattr(nn.layer, module_name)
        if inspect.isclass(ms_layer):
            MindFormerRegister.register_cls(
                ms_layer, MindFormerModuleType.BASE_LAYER)


register_ms_layer()
