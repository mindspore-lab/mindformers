import inspect

from mindspore import nn

from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_wrap(config: dict = None, default_args: dict = None,
               module_type: str = 'wrapper', class_name: str = None,
               *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        return XFormerRegister.get_instance_from_cfg(
            config, XFormerModuleType.WRAPPER, default_args=default_args)
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def build_runner_wrap(config: dict = None, default_args: dict = None, class_name: str = None,
                      *args, **kwargs):
    loss_scale = build_wrap(config.loss_scale, class_name=class_name, *args, **kwargs)
    if default_args is not None and isinstance(default_args, dict):
        if loss_scale is None:
            loss_scale = 1.0
        default_args.setdefault('scale_sense', loss_scale)
    wrapper = build_wrap(config.train_one_step, default_args=default_args, class_name=class_name, *args, **kwargs)
    return wrapper


def register_ms_wrap():
    """ register MindSpore builtin wrapper class. """
    for module_name in dir(nn.wrap):
        if module_name.startswith('__'):
            continue
        wrap = getattr(nn.wrap, module_name)
        if inspect.isclass(wrap):
            XFormerRegister.register_cls(
                wrap, XFormerModuleType.WRAPPER)


register_ms_wrap()
