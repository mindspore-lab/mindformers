# LLM数据在线加载

Mindformers大模型套件支持直接读取非mindrecord格式的数据，如json、parquet等，主要依赖TrainingDataLoader和SFTDataLoader实现，
其中是TrainingDataLoader主要用于预训练数据集的读取，SFTDataLoader主要用于微调数据集的读取。

## TrainingDataLoader

### 功能

主要用于预训练数据集的读取，也可用于与预训练数据集类似的评测数据集的读取。配合CausalLanguageModelDataset使用。支持token拼接，增加有效token的长度。支持自定义读取方式。

### 参数

- dataset_dir: 数据集的文件路径，支持具体的文件路径，也支持包含多个数据文件的目录。当配置为目录时，配合`file_format`参数筛选具体的文件。
- column_names: 创建的GeneratorDataset数据集中包含的列名。
- tokenizer: tokenizer配置参数, 可以是字典或者字符串，也可以直接配置tokenizer对象。
- dataset_name: 数据集名称。TrainingDataLoader内置了一些经典数据集的读取方式，目前仅支持`wikitext`数据集。
- is_align: 是否拼接对齐tokens，默认配置为True。当为True时，会将encoder之后的多个句子拼接对齐为`max_length`长度，超出部分将作为下一条样本。
- max_length: tokens的最大长度。当`is_align`参数设置为True时，即tokens需要拼接对齐时生效。默认配置为1025。
- text_col: 原始数据集中需要参与预训练的数据列名称。默认获取第一列数据。
- file_format: 数据文件的格式，支持json、jsonl、csv、tsv和parquet五种格式的数据。其中由于json格式的数据结构多样，
  仅支持类似[{k1:a1,k2:b1},...,{k1:an,k2:bn}]与{k1:[a1,...,an],k2:[b1,...,bn]}格式的两层嵌套的数据。
- read_function: 自定义的数据读取方法。此方法的入参为文件路径，返回值是一个字典，其中key代表列名，value为这一列的数据。
- shuffle: 是否打乱样本的顺序。
- samples_num: 样本总数，默认配置为10000。因为预训练的数据可能是海量的，并且可能存在tokes拼接的情况，实际的样本总数不容易确定，因此需要自行指定需要的样本总数。
  当已读取处理的样本数达到`samples_num`时，将停止读取。如果配置的`samples_num`超出实际的样本数量，为保持训练的连续性，将重新进行读取处理。
- skip_num: 跳过指定数量的样本。
- file_limit: 每次读取的文件数量。预训练数据可能包含很多同类型的文件，采用分批读取的方式，`file_limit`用于指定每个批次读取的文件数量。
- kwargs: 支持MindSpore的GeneratorDataset类的所有参数。

### 脚本启动配置方式

```yaml
data_loader:
    type: TrainingDataLoader
    dataset_dir: ""
    column_names: ["input_ids", "attention_mask"]
    tokenizer:
        type: GPT2Tokenizer
    max_length: 1025
    shuffle: True
    file_format: tokens
    dataset_name: wikitext
```

### API调用方式

```python
from mindformers import TrainingDataLoader
data_loader = TrainingDataLoader(dataset_dir="{your_path/wiki.train.tokens}",
                                 column_names=["input_ids", "attention_mask"],
                                 tokenizer={"type": "GPT2Tokenizer"},
                                 dataset_name="wikitext",
                                 file_format="tokens",
                                 max_length=1025,
                                 shuffle=True)
data_loader = data_loader.batch(1)
for item in data_loader:
    print(item)
    break
```

## SFTDataLoader

### 功能

主要用于微调数据集的读取，也可用于与微调数据集类似的评测数据集的读取。配合CausalLanguageModelDataset使用。支持自定义读取方式，支持自定义解析方式，支持多轮对话数据集。

### 参数

- dataset_dir: 数据集的文件路径，支持具体的文件路径，也支持包含parquet文件的目录。
- dataset_name: 数据集名称。SFTDataLoader内置了一些经典数据集是读取或解析方式，目前支持`alpaca`、`advertisegen`、`cola`、
  `imdb`、`sst-2`、`ag-news`、`tnews`、`squad`、`cmrc2018`、`ag-news`、`multi-round-chat`数据集，其中，`multi-round-chat`代表
  处理多轮对话的数据集，后文中有详细说明。
- file_format: 数据文件的格式，支持`json`、`jsonl`、`csv`、`tsv`和`parquet`五种格式的数据``。其中由于json格式的数据结构多样，
  仅支持类似[{k1:a1,k2:b1},...,{k1:an,k2:bn}]与{k1:[a1,...,an],k2:[b1,...,bn]}格式的两层嵌套的数据。
- column_names: 创建的GeneratorDataset数据集中包含的列名。
- tokenizer: tokenizer配置参数, 可以是字典或者字符串，也可以直接配置tokenizer对象。
- max_length: tokens的最大长度。当`is_align`参数设置为True时，即tokens需要拼接对齐时生效。默认配置为1025。  
- read_function: 此方法的入参为文件路径，返回值是一个字典，其中key代表列名，value为这一列的数据。
- map_function: 自定义的映射方法，可以将一个数据集的一行数据映射为一个新数据集的一行数据。此方法的入参是一个字典，包含数据集某一行的数据，
  可以根据key获取到相应列的值， 返回值也是一个字典，key代表新的列名，value代表相应的取值。额外的入参需要通过`map_function_kwargs`参数传入。
- map_function_kwargs: `map_function`的额外参数，字典格式，默认包含`tokenizer`和`max_length`参数，除此之外的参数需要显性传入。
- shuffle: 是否打乱样本的顺序。
- kwargs: 支持MindSpore的GeneratorDataset类的所有参数。

### 脚本启动配置方式

```yaml
data_loader:
    type: SFTDataLoader
    dataset_dir: ""
    column_names: ["input_ids"]
    tokenizer: GPT2Tokenizer
    max_length: 1025
    shuffle: True
    dataset_name: alpaca
    file_format: json
```

### API调用方式

```python
from mindformers import SFTDataLoader
data_loader = SFTDataLoader(dataset_dir="{your_path/alpaca_data.json}",
                            column_names=["input_ids"],
                            tokenizer="GPT2Tokenizer",
                            dataset_name="alpaca",
                            file_format="json",
                            max_length=1025,
                            shuffle=True)
data_loader = data_loader.batch(1)
for item in data_loader:
    print(item)
    break
```

### 处理多轮对话数据

#### 默认配置

SFTDataLoader支持读取和处理多轮对话的数据，当SFTDataLoader的入参`dataset_name`配置为`multi-round-chat`时，即为使用内置的多轮对话处理
方式。默认可以读取形如以下结构的数据（示例数据仅为数据集中的一个样本）：

```json
 {
  "id": "27684",
  "conversations": [
   {
    "from": "human",
    "value": "你好，请问你能帮我查一下明天的天气吗？\n"
   },
   {
    "from": "gpt",
    "value": "当然，你在哪个城市呢？\n"
   },
   {
    "from": "human",
    "value": "我在上海。\n"
   },
   {
    "from": "gpt",
    "value": "好的，根据天气预报，明天上海多云转阴，气温在20到25摄氏度之间。需要我帮你查询其他信息吗？"
   }
  ]
 }
```

#### 自定义配置

对于与示例数据结构相似，但字段不相同的多轮对话数据集，可以通过SFTDataLoader的入参`map_function_kwargs`进行适配，`dataset_name`配置为
`multi-round-chat`时，`map_function_kwargs`支持以下关键字参数：

- data_field: 对话数据所在的字段名称，默认为`conversations`。
- from_keyword: 代表对话语句来源的关键字，默认为`from`，用于区分对话的双方。
- value_keyword: 代表对话语句内容的关键字，默认为`value`，用于承载对话语句内容。
- user_role_name: 对话发起者，默认为`human`, 一般代表提问方。
- assistant_role_name: 对话协作者，默认为`gpt`，一般代表回答方。
- user_prompt: 对话发起者提示语，用于加在对话发起者的语句前面。无默认值，不指定则不添加。
- assistant_prompt: 对话协作者提示语，用于加在对话协作者的语句前面。无默认值，不指定则不添加。
- ignore_token_id: 计算label时使用，用于遮罩对话发起者或提问方的语句，默认为`-100`。

例如，以下的多轮对话数据，与【默认配置】章节中的数据在结构上相似，仅仅是部分字段名称不相同，将`map_function_kwargs`配置为
`{"data_field": "data", "value_keyword": "text", "assistant_role_name": "assistant"}`既可读取以下的多轮对话数据。

```json
 {
  "id": "73025",
  "data": [
   {
    "from": "human",
    "text": " 生成一首诗。\n"
   },
   {
    "from": "assistant",
    "text": " 好的，请给一个主题或者几个关键字。\n"
   },
   {
    "from": "human",
    "text": " 春雨纷纷，绿叶成阴。\n"
   },
   {
    "from": "assistant",
    "text": " 雨声潺潺湿衣巾，绿叶遮天隐栋梁。\n         一夜春雨长不息，驱散寒气渐放晴。\n         "
   }
  ]
 }
```

#### 自定义方法

对于与示例数据结构不相似的数据，内置的多轮对话处理方式不适用，需要自定义`map_function`自行处理，详情参考`map_function`参数介绍。

