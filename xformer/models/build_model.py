from xformer.tools.register import XFormerRegister, XFormerModuleType


def build_model(config: dict = None, default_args: dict = None,
                module_type: str = 'models', class_name: str = None,
                *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        if default_args is None:
            default_args = {}

        model_config = build_model_config(config.model_config, default_args=default_args)

        if model_config is not None:
            if default_args is not None:
                for key, value in default_args.items():
                    model_config.__setattr__(key, value)
                default_args = {}
            default_args.setdefault('config', model_config)

            return XFormerRegister.get_instance_from_cfg(
                config.arch, XFormerModuleType.MODELS, default_args=default_args)
        return None
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def build_tokenizer(
        config: dict = None, default_args: dict = None,
        module_type: str = 'tokenizer', class_name: str = None,
        *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        return XFormerRegister.get_instance_from_cfg(
            config, XFormerModuleType.TOKENIZER, default_args=default_args)
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def build_encoder(
        config: dict = None, default_args: dict = None,
        module_type: str = 'encoder', class_name: str = None,
        *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        cfg_encoders = config
        if not isinstance(cfg_encoders, list):
            return XFormerRegister.get_instance_from_cfg(
                cfg_encoders, XFormerModuleType.ENCODER, default_args=default_args)
        encoders = []
        for encoder in cfg_encoders:
            encoder_op = XFormerRegister.get_instance_from_cfg(
                encoder, XFormerModuleType.ENCODER)
            encoders.append(encoder_op)
        return encoders
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def build_head(
        config: dict = None, default_args: dict = None,
        module_type: str = 'head', class_name: str = None,
        *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        cfg_heads = config
        if not isinstance(cfg_heads, list):
            return XFormerRegister.get_instance_from_cfg(
                cfg_heads, XFormerModuleType.HEAD, default_args=default_args)
        heads = []
        for head in cfg_heads:
            head_op = XFormerRegister.get_instance_from_cfg(
                head, XFormerModuleType.HEAD)
            heads.append(head_op)
        return heads
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)


def build_model_config(
        config: dict = None, default_args: dict = None,
        module_type: str = 'config', class_name: str = None,
        *args, **kwargs):
    if config is None and class_name is None:
        return None
    if config is not None:
        if config.text_config is not None:
            config.text_config = build_model_config(config.text_config)
        if config.vision_config is not None:
            config.vision_config = build_model_config(config.vision_config)
        if config.head_config is not None:
            config.head_config = build_model_config(config.head_config)
        return XFormerRegister.get_instance_from_cfg(
            config, XFormerModuleType.CONFIG, default_args=default_args)
    return XFormerRegister.get_instance(module_type, class_name, *args, **kwargs)
