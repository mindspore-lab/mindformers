def check_mim_model_config(config):
    if config.model.arch is not None and config.model.encoder is None:
        config.model.arch.image_size = config.runner_config.image_size
        config.model.arch.batch_size = config.runner_config.batch_size
    if config.model.encoder is not None:
        config.model.encoder.image_size = config.runner_config.image_size
        config.model.encoder.batch_size = config.runner_config.batch_size
