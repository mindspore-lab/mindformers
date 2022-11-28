"""Masked Image Modeling Dataset."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from .dataloader import build_dataset_loader
from .mask import build_mask
from .transforms import build_transforms
from .sampler import build_sampler
from .base_dataset import BaseDataset


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class MIMDataset(BaseDataset):
    """Masked Image Modeling Dataset."""
    def __new__(cls, dataset_config: dict = None):
        super().__init__(dataset_config)
        dataset = build_dataset_loader(dataset_config.data_loader)
        transforms = build_transforms(dataset_config.transforms)
        mask = build_mask(dataset_config.mask_policy)
        sampler = build_sampler(dataset_config.sampler)

        if sampler is not None:
            dataset = dataset.use_sampler(sampler)

        if transforms is not None:
            for column in dataset_config.input_columns:
                dataset = dataset.map(
                    input_columns=column,
                    operations=transforms,
                    num_parallel_workers=dataset_config.num_parallel_workers,
                    python_multiprocessing=dataset_config.python_multiprocessing)

        if mask is not None:
            dataset = dataset.map(
                operations=mask,
                input_columns=dataset_config.input_columns,
                column_order=dataset_config.column_order,
                output_columns=dataset_config.output_columns,
                num_parallel_workers=dataset_config.num_parallel_workers,
                python_multiprocessing=dataset_config.python_multiprocessing)
        dataset = dataset.batch(dataset_config.batch_size, drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset
