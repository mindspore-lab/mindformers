"""Build Loss API."""
import inspect

from mindspore import nn

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_loss(
        config: dict = None, default_args: dict = None,
        module_type: str = 'loss', class_name: str = None, **kwargs):
    """Build Loss API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.LOSS, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_loss():
    """ register MindSpore builtin loss class. """
    for module_name in dir(nn.loss):
        if module_name.startswith('__'):
            continue
        loss = getattr(nn.loss, module_name)
        if inspect.isclass(loss):
            MindFormerRegister.register_cls(
                loss, MindFormerModuleType.LOSS)


register_ms_loss()
