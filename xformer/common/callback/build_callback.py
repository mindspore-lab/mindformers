import inspect

from mindspore.train import callback

from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_callback(
        config: dict = None, default_args: dict = None,
        module_type: str = 'callback', class_name: str = None,
        *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        cfg_callback = config
        if not isinstance(cfg_callback, list):
            return XFormerRegister.get_instance_from_cfg(
                cfg_callback, XFormerModuleType.CALLBACK, default_args=default_args)
        callbacks = []
        for callback_type in cfg_callback:
            callback_op = XFormerRegister.get_instance_from_cfg(
                callback_type, XFormerModuleType.CALLBACK)
            callbacks.append(callback_op)
        return callbacks
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def register_ms_cb():
    """ register MindSpore builtin LR class. """
    for module_name in dir(callback):
        if module_name.startswith('__'):
            continue
        cb = getattr(callback, module_name)
        if inspect.isclass(cb):
            XFormerRegister.register_cls(
                cb, XFormerModuleType.CALLBACK)


register_ms_cb()
