"""Build Trainer API."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_trainer(
        config: dict = None, default_args: dict = None,
        module_type: str = 'trainer', class_name: str = None, **kwargs):
    """Build trainer API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.TRAINER, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)
