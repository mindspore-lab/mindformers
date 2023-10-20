# 读取非mindrecord格式的数据

Mindformers大模型套件支持直接读取非mindrecord格式的数据，如json、parquet等，主要依赖TrainingDataLoader和SFTDataLoader实现，
其中是TrainingDataLoader主要用于预训练数据集的读取，SFTDataLoader主要用于微调数据集的读取。

## TrainingDataLoader

### 功能

主要用于预训练数据集的读取，也可用于与预训练数据集类似的评测数据集的读取。支持token拼接，增加有效token的长度。支持自定义读取方式。

### 参数

- dataset_dir: 数据集的文件路径，支持具体的文件路径，也支持包含多个数据文件的目录。当配置为目录时，配合`file_format`参数筛选具体的文件。
- dataset_name: 数据集名称。TrainingDataLoader内置了一些经典数据集的读取方式，目前仅支持`wikitext`数据集。
- max_length: tokens的最大长度。当`is_align`参数设置为True时，即tokens需要拼接对齐时生效。默认配置为1025。
- is_align: 是否拼接对齐tokens，默认配置为True。当为True时，会将encoder之后的多个句子拼接对齐为`max_length`长度，超出部分将作为下一条样本。
- tokenizer: tokenizer配置参数。
- text_col: 原始数据集中需要参与预训练的数据列名称。默认获取第一列数据。
- file_format: 数据文件的格式，支持json、jsonl、csv、tsv和parquet五种格式的数据。其中由于json格式的数据结构多样，
  仅支持类似[{k1:a1,k2:b1},...,{k1:an,k2:bn}]与{k1:[a1,...,an],k2:[b1,...,bn]}格式的两层嵌套的数据。
- customized_reader: 自定义的数据读取方法。此方法的入参为文件路径，返回值是一个pyarrow的Table对象。
- shuffle: 是否打乱样本的顺序。
- samples_num: 样本总数，默认配置为10000。因为预训练的数据可能是海量的，并且可能存在tokes拼接的情况，实际的样本总数不容易确定，因此需要自行指定需要的样本总数。
  当已读取处理的样本数达到`samples_num`时，将停止读取。如果配置的`samples_num`超出实际的样本数量，为保持训练的连续性，将重新进行读取处理。
- skip_num: 跳过指定数量的样本。
- file_limit: 每次读取的文件数量。预训练数据可能包含很多同类型的文件，采用分批读取的方式，`file_limit`用于指定每个批次读取的文件数量。
- kwargs: 支持MindSpore的GeneratorDataset类的所有参数。

### 脚本启动配置方式

```yaml
tokenizer: &tokenizer
    type: GPT2Tokenizer
    max_length: 1025
data_loader:
    type: TrainingDataLoader
    dataset_dir: ""
    shuffle: True
    file_format: tokens
    tokenizer: *tokenizer
    dataset_name: wikitext
```

### API调用方式

```python
from mindformers import TrainingDataLoader
data_loader = TrainingDataLoader(dataset_dir="{your_path/wiki.train.tokens}",
                                 dataset_name="wikitext",
                                 file_format="tokens",
                                 max_length=1025,
                                 tokenizer={"type": "GPT2Tokenizer"},
                                 shuffle=True)
data_loader = data_loader.batch(1)
for item in data_loader:
    print(item)
    break
```

## SFTDataLoader

### 功能

主要用于微调数据集的读取，也可用于与微调数据集类似的评测数据集的读取。支持自定义读取方式，支持自定义解析方式。

### 约束

1、暂不支持多轮对话数据集

### 参数

- dataset_dir: 数据集的文件路径，支持具体的文件路径，也支持包含parquet文件的目录。
- dataset_name: 数据集名称。SFTDataLoader内置了一些经典数据集是读取或解析方式，目前支持`alpaca`、`advertisegen`、`cola`、
  `imdb`、`sst-2`、`ag-news`、`tnews`、`squad`、`cmrc2018`、`ag-news`数据集。
- file_format: 数据文件的格式，支持json、jsonl、csv、tsv和parquet五种格式的数据。其中由于json格式的数据结构多样，
  仅支持类似[{k1:a1,k2:b1},...,{k1:an,k2:bn}]与{k1:[a1,...,an],k2:[b1,...,bn]}格式的两层嵌套的数据。
- customized_reader: 自定义的数据读取方法。此方法的入参为文件路径，返回值是一个pyarrow的Table对象。
- customized_parser: 自定义的数据解析方法。此方法的入参是一个字典，包含数据集某一行的数据，可以根据key获取到相应列的值，
  返回值有三个，分别代表prompt、answer、label的取值。内置默认解析方法会返回入参的前三个值分别作为prompt、answer、label的取值。
- shuffle: 是否打乱样本的顺序。
- kwargs: 支持MindSpore的GeneratorDataset类的所有参数。

### 脚本启动配置方式

```yaml
data_loader:
    type: SFTDataLoader
    dataset_dir: ""
    shuffle: True
    dataset_name: alpaca
    file_format: json
```

### API调用方式

```python
from mindformers import SFTDataLoader
data_loader = SFTDataLoader(dataset_dir="{your_path/alpaca_data.json}",
                            dataset_name="alpaca",
                            file_format="json",
                            shuffle=True)
data_loader = data_loader.batch(1)
for item in data_loader:
    print(item)
    break
```
