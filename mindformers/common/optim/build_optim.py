"""Build Optimizer API."""
import inspect

from mindspore import nn

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_optim(
        config: dict = None, default_args: dict = None,
        module_type: str = 'optimizer', class_name: str = None, **kwargs):
    """Build Optimizer API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.OPTIMIZER, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_optim():
    """ register MindSpore builtin optimizer class. """
    for module_name in dir(nn.optim):
        if module_name.startswith('__'):
            continue
        optim = getattr(nn.optim, module_name)
        if inspect.isclass(optim):
            MindFormerRegister.register_cls(
                optim, MindFormerModuleType.OPTIMIZER)


register_ms_optim()
