def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def check_dataset_config(config, dataset):
    data_size = dataset.get_dataset_size()
    new_epochs = config.runner_config.epochs
    if config.runner_config.per_epoch_size and config.runner_config.sink_mode:
        config.runner_config.epochs = int((data_size / config.runner_config.per_epoch_size) * new_epochs)
    else:
        config.runner_config.per_epoch_size = data_size

    config.data_size = data_size
    config.logger.info("Will be Training epochs:{}， sink_size:{}".format(
        config.runner_config.epochs, config.runner_config.per_epoch_size))
    config.logger.info("Create training dataset finish, dataset size:{}".format(data_size))