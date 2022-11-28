"""Build Core API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_core(config: dict = None, default_args: dict = None,
               module_type: str = 'core', class_name: str = None, **kwargs):
    """Build core API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.CORE, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
