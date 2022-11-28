"""Build Callback API."""
import inspect

from mindspore.train import callback

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_callback(
        config: dict = None, default_args: dict = None,
        module_type: str = 'callback', class_name: str = None, **kwargs):
    """Build callback API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        cfg_callback = config
        if not isinstance(cfg_callback, list):
            return MindFormerRegister.get_instance_from_cfg(
                cfg_callback, MindFormerModuleType.CALLBACK, default_args=default_args)
        callbacks = []
        for callback_type in cfg_callback:
            callback_op = MindFormerRegister.get_instance_from_cfg(
                callback_type, MindFormerModuleType.CALLBACK)
            callbacks.append(callback_op)
        return callbacks
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_cb():
    """ register MindSpore builtin LR class. """
    for module_name in dir(callback):
        if module_name.startswith('__'):
            continue
        monitor = getattr(callback, module_name)
        if inspect.isclass(monitor):
            MindFormerRegister.register_cls(
                monitor, MindFormerModuleType.CALLBACK)


register_ms_cb()
