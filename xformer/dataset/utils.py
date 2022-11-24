def check_mim_dataset_config(config):
    if config.train_dataset is not None:
        config.pretrain_dataset.seed = config.seed
        config.pretrain_dataset.auto_tune = config.auto_tune
        config.pretrain_dataset.filepath_prefix = config.filepath_prefix
        config.pretrain_dataset.autotune_per_step = config.autotune_per_step
        config.pretrain_dataset.profile = config.profile
        config.pretrain_dataset.batch_size = config.runner_config.batch_size
        config.pretrain_dataset_task.dataset_config = config.pretrain_dataset

    if config.eval_dataset is not None:
        pass

    if config.predict_dataset is not None:
        pass
