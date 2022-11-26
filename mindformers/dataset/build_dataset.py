"""Build Dataset API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_dataset(
        config: dict = None, default_args: dict = None,
        module_type: str = 'dataset', class_name: str = None, **kwargs):
    """Build dataset API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.DATASET, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
