# 使用datasets进行数据集预处理和加载

## 目的

接入openMind仓库、HuggingFace仓库，在线加载数据集，扩大数据集来源。

使用dataset增强数据集加载和处理能力。

## 环境准备

1、配置远程仓库

环境变量`HF_ENDPOINT`可以控制开源社区huggingFace实际使用的远程仓库，未配置时默认为`https://huggingFace.co`，针对国内环境，需要配置成镜像地址`https://hf-mirror.com`
环境变量`OPENMIND_HUB_ENDPOINT`可以控制开源社区OpenMind实际使用的远程仓库，未配置时默认为`https://telecom.openmind.cn`。

2、安装依赖

```shell
git clone https://gitee.com/openmind-ai/openmind-hub.git
cd openmind-hub
pip install -e .
cd ..
pip install datasets==2.18.0
git clone https://gitee.com/openmind-ai/openmind-extension-for-datasets.git
cd openmind-extension-for-datasets
pip install -e .
cd ..
```

## 对接数据集

使用DataLoader加载各种数据集
1、对于mindrecord格式的离线数据，如json、parquet等，支持使用MindDataset类型的数据加载方式，参考如下yaml配置

```yaml
data_loader:
  type: MindDataset
  dataset_dir: "本地数据集路径"
  shuffle: False
```

2、对于非mindrecord格式的数据，提供了通用的commonDataLoader和自定义XXXDataLoader两种方式加载，参考如下yaml配置

```yaml
data_loader:
  type: CommonDataLoader
  dataset_path: "om:远端数据集路径"
  shuffle: False
  split: "train"

data_loader:
  type: XXXDataLoader
  dataset_path: "om:远端数据集路径"
  shuffle: False
  split: "train"
```

### XXXDataLoader自定义方式加载数据集

自定义DataLoader主要是用于加载离线数据，或者远端数据集，子类需要继承基类BaseDataLoader

```python
@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class XXXDataLoader(BaseDataLoader):
  # 自定义数据处理逻辑
  # return GeneratorDataset
```

### CommonDataLoader方式加载数据集

#### 功能

CommonDataLoader定义了通用的流程步骤：1、加载远端数据集（支撑huggingFace、openMind）得到开源的datasets数据集；2、自定义数据处理DataHandler模块（可选：支持用户对加载到的数据集做定制逻辑转换）；3、开源的datasets转换为ms.datasets

#### 参数

- type: 必填，数据加载的处理方式，支持3种方式：MindDataset、CommonDataLoader、自定义XXXDataLoader
- dataset_path： 可选，对接远端数据集路径，支撑3种方式：hf:开头的代表对接huggingFace上的数据集，om: 开头的代表对接openMind上的数据集，local：开头代表本地数据集，非以上3种开头的，默认是对接openMind数据集
- dataset_dir: 可选，本地数据集的路径，此参数和dataset_path必须有一个是非空的
- shuffle: 必填，数据集是否打乱
- split: 可选，子数据集的名称，默认加载train集
- data_files: 可选，数据文件列表
- token: 可选，加载远端私有数据集时鉴权使用
- handler：可选，自定义数据处理，配套type为CommonDataLoader时使用

#### 自定义datahandler

用户可以使用自定义的dataHandler逻辑，对加载到的远端数据集进行数据预处理定制逻辑

##### 参数

- type: 必填，自定义数据处理handler名称，自定义handler必须继承BaseInstructDataHandler
- tokenizer_name: 必填，使用的tokenizer分词器名称
- seq_length: 必填，序列长度
- output_columns: 必填，数据预处理后输出的数据列
- prompt_key: 可选，增加prompt处理后数据列名称
- tokenizer: 可选，tokenizer配置参数, 可以是字典或者字符串，也可以直接配置tokenizer对象。

##### 开发样例

自定义dataHandler一般放在mindformers/dataset/handler目录下，自定义的需要继承抽象基类base_handler，需要实现format_func、tokenize_func两个方法，可以参考alpaca_handler.py、deepseek_handler.py

```python
@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class AlpacaInstructDataHandler(BaseInstructDataHandler):
```

format_func用于实现如何从原始数据中，转换成你所需要的数据格式，

```python
def format_func(self, data):
    # 自定义处理逻辑
```

tokenize_func方法用于把处理后的数据进行按自定义分词

```python
def tokenize_func(self, messages):
  # 自定义处理逻辑
```

## alpaca数据集示例

### 训练流程直接从远端仓库加载

修改任务配置文件`finetune_llama2_7b.yaml`。

修改如下参数：

```yaml
train_dataset:
  input_columns: &input_columns ["input_ids", "labels"]
  data_loader:
    type: CommonDataLoader
    dataset_dir: ""
    shuffle: True
    split: "train"
    dataset_path: "AI_Connect/alpaca"
    input_columns: *input_columns
    handler:
      type: AlpacaInstructDataHandler
      tokenizer_name: llama2_13b
      seq_length: 4096
      prompt_key: "conversations"
      output_columns: *input_columns

# 参数说明
input_columns: 必填，输入的数据的列名
data_loader.type: 必填，数据加载处理的类名
data_loader.shuffle: 必填，数据集是否打乱
data_loader.dataset_path: 可选，加载数据集的远端仓库名称（hf:开头的为对接huggingFace数据集，om：开头的为对接OpenMind数据集，local：开头的为对接本地离线数据集，不以以上开头的默认对接的为openMind）
data_loader.dataset_dir: 可选，加载数据集使用的本地路径
data_loader.data_dir: 可选，数据集目录路径
data_loader.data_files: 可选，数据集文件路径
data_loader.split: 可选，数据集子集，默认加载train集
data_loader.token: 可选，加载私有数据集或上传到私人仓库时必须
data_loader.handler: 可选，数据预处理类配置，为空时不做数据处理
data_loader.handler.type: 数据预处理类的类名
data_loader.handler.tokenizer_name: 分词器名称
data_loader.handler.seq_length: 序列长度
data_loader.handler.prompt_key: 可选，增加prompt处理后数据列名称
data_loader.handler.output_columns: 可选，数据预处理后输出的数据列
```

## ADGEN数据集示例

### 训练流程直接从远端仓库加载

修改任务配置文件`run_glm3_6b_finetune_2k_800T_A2_64G.yaml`。

修改如下参数：

```yaml
train_dataset: &train_dataset
  data_loader:
    type: ADGenDataLoader
    dataset_dir: "/path/to/AdvertiseGen/train.json"
    shuffle: True
    phase: "train"
    version: 3
    origin_columns: ["content", "summary"]
    dataset_path: "zhangyifei/ADGEN"

# 参数说明
data_loader.type: 必填，数据加载处理的类名
data_loader.dataset_path: 可选，加载数据集的远端仓库名称（hf:开头的为对接huggingFace数据集，om：开头的为对接OpenMind数据集，local：开头的为对接本地离线数据集，不以以上开头的默认对接的为openMind）
data_loader.dataset_dir: 可选，加载数据集使用的本地路径
data_loader.shuffle: 必填，数据集是否打乱
data_loader.data_dir: 可选，数据集目录路径
data_loader.data_files: 可选，数据集文件路径
data_loader.phase: 可选，数据集子集，默认加载train集
data_loader.token: 可选，加载私有数据集或上传到私人仓库时必须
```

## Qwen-VL数据集示例

### 训练流程直接从远端仓库加载

修改任务配置文件`finetune_qwenvl_9.6b_bf16.yaml`。

修改如下参数：

```yaml
train_dataset: &train_dataset
  data_loader:
    type: QwenVLDataLoader
    dataset_dir: "/location/of/images"
    dataset_path: "zhangyifei/qwenvl"
    annotation_file: "conversation_file.json"
    column_names: [ "image", "text" ]
    shuffle: True
    extra_kwargs:
      max_img_len: 1
      map_function_kwargs:
        user_role_name: user
        assistant_role_name: assistant

# 参数说明
data_loader.type: 必填，数据加载处理的类名
data_loader.dataset_path: 可选，加载数据集的远端仓库名称（hf:开头的为对接huggingFace数据集，om：开头的为对接OpenMind数据集，local：开头的为对接本地离线数据集，不以以上开头的默认对接的为openMind）
data_loader.shuffle: 必填，数据集是否打乱
data_loader.dataset_dir: 必填，加载图片数据集使用的本地路径
data_loader.data_files: 可选，数据集文件路径
data_loader.split: 可选，数据集子集，默认加载train集
data_loader.token: 可选，加载私有数据集或上传到私人仓库时必须
data_loader.extra_kwargs: 必填，扩展参数
```

## code_alpaca(DeepSeek-Coder)数据集示例

### 训练流程直接从远端仓库加载

修改任务配置文件`finetune_deepseek_33b.yaml`。

修改如下参数：

```yaml
train_dataset: &train_dataset
  data_loader:
    type: CommonDataLoader
    dataset_dir: ""
    shuffle: False
    dataset_path: 'AI_Connect/CodeAlpaca-20K'
    split: 'train'
    handler:
      type: DeepSeekInstructDataHandler
      tokenizer_name: deepseek_33b
      seq_length: 4096
      prompt_key: "conversations"
      output_columns: ["input_ids", "labels"]
      tokenizer:
        unk_token: None
        bos_token: '<｜begin▁of▁sentence｜>'
        eos_token: '<|EOT|>'
        pad_token: '<｜end▁of▁sentence｜>'
        vocab_file: None
        tokenizer_file: "path/to/deepseek/tokenizer.json"  # tokenizer.json
        type: LlamaTokenizerFast

# 参数说明
data_loader.type: 必填，数据加载处理的类名
data_loader.dataset_path: 可选，加载数据集的远端仓库名称（hf:开头的为对接huggingFace数据集，om：开头的为对接OpenMind数据集，local：开头的为对接本地离线数据集，不以以上开头的默认对接的为openMind）
data_loader.shuffle: 必填，数据集是否打乱
data_loader.dataset_dir: 必填，加载图片数据集使用的本地路径
data_loader.data_files: 可选，数据集文件路径
data_loader.split: 可选，数据集子集，默认加载train集
data_loader.token: 可选，加载私有数据集或上传到私人仓库时必须
data_loader.handler: 可选，数据预处理类配置，为空时不做数据处理
data_loader.handler.type: 数据预处理类的类名
data_loader.handler.tokenizer_name: 分词器名称
data_loader.handler.seq_length: 序列长度
data_loader.handler.prompt_key: 可选，增加prompt处理后数据列名称
data_loader.handler.tokenizer: 必填，tokenizer的详细配置
```