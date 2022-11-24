import inspect

from mindspore.dataset import samplers as sp

from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_sampler(config: dict = None, default_args: dict = None,
                  module_type: str = 'dataset_sampler', class_name: str = None,
                  *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        return XFormerRegister.get_instance_from_cfg(
            config, XFormerModuleType.DATASET_SAMPLER, default_args=default_args)
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def register_ms_samplers():
    """ register MindSpore builtin transforms class. """
    for module_name in dir(sp):
        if module_name.startswith('__'):
            continue

        samplers = getattr(sp, module_name)
        if inspect.isclass(samplers):
            XFormerRegister.register_cls(samplers, XFormerModuleType.DATASET_SAMPLER)


register_ms_samplers()
