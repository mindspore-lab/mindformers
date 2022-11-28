"""Build Processor API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_processor(
        config: dict = None, default_args: dict = None,
        module_type: str = 'processor', class_name: str = None, **kwargs):
    """Build processor API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.PROCESSOR, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
