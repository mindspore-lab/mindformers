"""Build Sampler API."""
import inspect

from mindspore.dataset import samplers as sp

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_sampler(
        config: dict = None, default_args: dict = None,
        module_type: str = 'dataset_sampler', class_name: str = None, **kwargs):
    """Build sampler API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.DATASET_SAMPLER, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_samplers():
    """ register MindSpore builtin transforms class. """
    for module_name in dir(sp):
        if module_name.startswith('__'):
            continue

        samplers = getattr(sp, module_name)
        if inspect.isclass(samplers):
            MindFormerRegister.register_cls(samplers, MindFormerModuleType.DATASET_SAMPLER)


register_ms_samplers()
