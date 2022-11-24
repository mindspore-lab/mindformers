import inspect

from mindspore import dataset as ds

from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_dataset_loader(config: dict = None, default_args: dict = None,
                         module_type: str = 'dataset_loader', class_name: str = None,
                         *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        return XFormerRegister.get_instance_from_cfg(
            config, XFormerModuleType.DATASET_LOADER, default_args=default_args)
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def register_ms_dataset_loader():
    """ register MindSpore builtin dataset loader class. """
    for module_name in dir(ds):
        if module_name.startswith('__'):
            continue
        dataset = getattr(ds, module_name)
        if inspect.isclass(dataset):
            XFormerRegister.register_cls(
                dataset, XFormerModuleType.DATASET_LOADER)


register_ms_dataset_loader()
