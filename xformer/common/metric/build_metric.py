import inspect

from mindspore import nn

from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_metric(config: dict = None, default_args: dict = None,
                 module_type: str = 'metric', class_name: str = None,
                 *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        return XFormerRegister.get_instance_from_cfg(
            config, XFormerModuleType.METRIC, default_args=default_args)
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def register_ms_mt():
    """ register MindSpore builtin LR class. """
    for module_name in dir(nn.metrics):
        if module_name.startswith('__'):
            continue
        mt = getattr(nn.metrics, module_name)
        if inspect.isclass(mt):
            XFormerRegister.register_cls(
                mt, XFormerModuleType.METRIC)


register_ms_mt()
