"""Build DataLoader API."""
import inspect

from mindspore import dataset as ds

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_dataset_loader(
        config: dict = None, default_args: dict = None,
        module_type: str = 'dataset_loader', class_name: str = None, **kwargs):
    """Build dataset loader API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.DATASET_LOADER, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_dataset_loader():
    """ register MindSpore builtin dataset loader class. """
    for module_name in dir(ds):
        if module_name.startswith('__'):
            continue
        dataset = getattr(ds, module_name)
        if inspect.isclass(dataset):
            MindFormerRegister.register_cls(
                dataset, MindFormerModuleType.DATASET_LOADER)


register_ms_dataset_loader()
