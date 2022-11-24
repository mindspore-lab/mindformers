import inspect

from mindspore.nn import transformer

from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_module(config: dict = None, default_args: dict = None,
                 module_type: str = 'modules', class_name: str = None,
                 *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        return XFormerRegister.get_instance_from_cfg(
            config, XFormerModuleType.MODULES, default_args=default_args)
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def register_ms_transformer():
    """ register MindSpore builtin xformer class. """
    for module_name in dir(transformer):
        if module_name.startswith('__'):
            continue
        tf = getattr(transformer, module_name)
        if inspect.isclass(tf):
            XFormerRegister.register_cls(
                tf, XFormerModuleType.MODULES)


register_ms_transformer()
