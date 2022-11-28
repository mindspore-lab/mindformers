"""Dataset Utils."""


def check_dataset_config(config):
    """Check dataset config."""
    if config.train_dataset is not None:
        config.train_dataset.seed = config.seed
        config.train_dataset.auto_tune = config.auto_tune
        config.train_dataset.filepath_prefix = config.filepath_prefix
        config.train_dataset.autotune_per_step = config.autotune_per_step
        config.train_dataset.profile = config.profile
        config.train_dataset.batch_size = config.runner_config.batch_size
        config.train_dataset_task.dataset_config = config.train_dataset

    if config.eval_dataset is not None:
        pass

    if config.predict_dataset is not None:
        pass
