"""Build Metric API."""
import inspect

from mindspore import nn

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_metric(
        config: dict = None, default_args: dict = None,
        module_type: str = 'metric', class_name: str = None, **kwargs):
    """Build Metric API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.METRIC, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_mt():
    """ register MindSpore builtin LR class. """
    for module_name in dir(nn.metrics):
        if module_name.startswith('__'):
            continue
        ms_metric = getattr(nn.metrics, module_name)
        if inspect.isclass(ms_metric):
            MindFormerRegister.register_cls(
                ms_metric, MindFormerModuleType.METRIC)


register_ms_mt()
