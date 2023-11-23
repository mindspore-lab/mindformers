# 多数据源在线加载

多数据源在线加载是提供给用户加载多个非`MindRecord`类型的数据源的能力，通过使用`MultiSourceDataLoader`通过对`MindFormers`
已有的`DataLoader`进行聚合，得到一个新的`DataLoader`，
而新的`DataLoader`可以被`MindFormers`上已有的`Dataset`加载，从而实现无侵入支持多数据源加载数据集。

## 配置方式

```yaml
train_dataset: &train_dataset
  data_loader:
    type: MultiSourceDataLoader
    samples_count: 1000
    dataset_ratios: [ 0.1, 0.15, 0.35, 0.4 ]
    # nums_per_dataset: [100, 150, 350, 400]
    sub_data_loader_args:
      stage: "train"
      column_names: [ "image", "text" ]
    sub_data_loader:
      - type: Flickr8kDataLoader
        dataset_dir: "./checkpoint_download/Flickr8k_0"
      - type: Flickr8kDataLoader
        dataset_dir: "./checkpoint_download/Flickr8k_1"
      - type: Flickr8kDataLoader
        dataset_dir: "./checkpoint_download/Flickr8k_2"
      - type: Flickr8kDataLoader
        dataset_dir: "./checkpoint_download/Flickr8k_3"
    shuffle: False
    shuffle_buffer_size: 320

    # 与原配置一致，如seed，transforms, text_transforms, tokenizer等，此处省略不写

train_dataset_task:
  type: ContrastiveLanguageImagePretrainDataset
  dataset_config: *train_dataset
```

## 参数

- samples_count: 指定从所有数据集中加载的总数据量，与参数`dataset_ratios`配合使用

- dataset_ratios: 指定从各个数据集中加载数量在`samples_count`中的占比

- nums_per_dataset: 指定从各个数据集中的加载数量，优先级低于`samples_count`+`dataset_ratios`的配置方式

- sub_data_loader: 指定子`DataLoader`配置，配置方式同原`DataLoader`的配置一致，子`DataLoader`中`shuffle`配置无效，由`MultiSourceDataLoader`的`shuffle`选项统一控制

- sub_data_loader_args: 子`DataLoader`的共同配置项，避免在`sub_data_loader`中填写冗余参数

- shuffle: 各个数据集的随机策略，与`MindRecord`的`shuffle`一致，并支持传入`bool`值，当值为`True`时，按`global`的方式进行shuffle，当值为`False`时，不对数据进行任何shuffle，具体如下
    - `global`: 对所有子数据集进行全局shuffle
    - `infile`: 子数据集内数据进行shuffle，子数据集顺序按照`sub_data_loader`的顺序
    - `files`: 数据集内数据不进行shuffle，对子数据集的顺序进行shuffle  

- shuffle_buffer_size: 各子数据集shuffle时的buffer大小，可不配置，默认为320

**注：** 当使用`MindSpore`原生实现的`data_loader`时，需要在`sub_data_loader_args`中添加相应的`column_names`，例如使用`ImageFolderDataset`时需要添加`"column_names": ["image", "label"]`， 原因是`MultiSourceDataLoader`通过使用`GenerateDataset`对子`data_loader`进行聚合，初始化`GenerateDataset`时需要指定该参数。

## 支持列表

`MultiSourceDataLoader`支持如下`Dataset`中加载对应的DataLoader:

- ZeroShotImageClassificationDataset
    - Cifar100DataLoader
    - Flickr8kDataLoader

- QuestionAnsweringDataset
    - SQuADDataLoader

- MIMDataset
    - ImageFolderDataset

- ImageCLSDataset
    - ImageFolderDataset

- ContrastiveLanguageImagePretrainDataset
    - Flickr8kDataLoader

- TokenClassificationDataset
    - CLUENERDataLoader