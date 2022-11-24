from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_core(config: dict = None, default_args: dict = None,
               module_type: str = 'core', class_name: str = None,
               *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        return XFormerRegister.get_instance_from_cfg(
            config, XFormerModuleType.CORE, default_args=default_args)
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)
