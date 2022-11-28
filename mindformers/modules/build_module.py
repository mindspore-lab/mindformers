"""Build Module API."""
import inspect

from mindspore.nn import transformer

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_module(
        config: dict = None, default_args: dict = None,
        module_type: str = 'modules', class_name: str = None, **kwargs):
    """Build module API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.MODULES, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_transformer():
    """ register MindSpore builtin mindformers class. """
    for module_name in dir(transformer):
        if module_name.startswith('__'):
            continue
        ms_transformer = getattr(transformer, module_name)
        if inspect.isclass(ms_transformer):
            MindFormerRegister.register_cls(
                ms_transformer, MindFormerModuleType.MODULES)


register_ms_transformer()
