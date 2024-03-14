# WizardCoder

## 模型描述

WizardCoder是由WizardLM团队推出了一个新的指令微调代码大模型，打破了闭源模型的垄断地位，超越了闭源大模型Anthropic Claude和谷歌的Bard。WizardCoder大幅度地提升了开源模型的SOTA水平，创造了惊人的进步，提高了22.3%的性能，成为了开源领域的新时代引领者。
WizardCoder完全开源可商用，基于 Transformer 结构，上下文窗口长度为 2048，参数量为150亿。 本仓库提供了WizardCoder-15B预训练模型。

## 仓库介绍

`WizardCoder` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`research/wizardcoder`

    ```bash
    wizardcoder
        ├── wizardcoder_tokenizer.py       # tokenizer
        ├── wizardcoder.py                 # 15B模型实现
        └── wizardcoder_modules.py         # self-attention模块实现
    ```

2. 模型配置：`research/wizardcoder`

    ```bash
    wizardcoder
        └── run_wizardcoder.yaml           # 15B全量微调Atlas 800T A2启动配置
    ```

3. 数据处理脚本和任务启动脚本：`research/wizardcoder`

    ```bash
    wizardcoder
        ├── wizardcoder_preprocess.py      # wizardcoder数据集预处理脚本
        └── run_wizardcoder.py             # wizardcoder高阶接口使用脚本
    ```

4. 评测脚本：`research/wizardcoder`

    ```bash
    wizardcoder
        ├── export_wizardcoder_inc.py      # 生成增量推理的mindir文件
        └── inference_wizardcoder_mslite.py     # 生成mindspore lite的推理性能结果
        └── inference_wizardcoder_pytorch.py    # 生成pytorch的推理性能结果
    ```

注：使用inference_wizardcoder_mslite.py测试推理速度时，需要修改mindformers源码中的`mindformers/inference/infers/text_generator_infer.py`文件，这样才可以打印出推理速度。
修改如下：

```python
class TextGeneratorInfer(BaseInfer):
    """
    Text generator infer implement class.
    """
    # pylint: disable=W0221
    def infer(self,
              inputs: Union[str, List[str]],
              do_sample: bool = False,
              top_k: int = 1,
              top_p: float = 1.0,
              temperature: float = 1.0,
              repetition_penalty: float = 1.0,
              eos_token_id: int = 2,
              pad_token_id: int = 0,
              max_length: int = 256,
              is_sample_acceleration: bool = False,
              add_special_tokens: bool = False,
              streamer: Optional[BaseStreamer] = None,
              **kwargs):
        # input_ids = self.preprocess(inputs, add_special_tokens)
        # output_ids = self.generate(input_ids, do_sample, top_k, top_p, temperature,
        #                            repetition_penalty, eos_token_id, pad_token_id,
        #                            max_length, is_sample_acceleration, streamer, **kwargs)
        # outputs = self.postprocess(output_ids)
        # return outputs
        start_time_with_tokenizer = time.time()
        input_ids = self.preprocess(inputs, add_special_tokens)
        start_time_no_tokenizer = time.time()
        output_ids = self.generate(input_ids, do_sample, top_k, top_p, temperature,
                        repetition_penalty, eos_token_id, pad_token_id,
                        max_length, is_sample_acceleration, streamer, **kwargs)
        end_time_no_tokenizer = time.time()
        outputs = self.postprocess(output_ids)
        end_time_with_tokenizer = time.time()
        elapsed_time_with_tokenizer = end_time_with_tokenizer - start_time_with_tokenizer
        elapsed_time_no_tokenizer = end_time_no_tokenizer - start_time_no_tokenizer
        generate_length = sum([len(item) for item in output_ids]) - sum([len(item) for item in input_ids])
        return outputs, generate_length, elapsed_time_with_tokenizer, elapsed_time_no_tokenizer
```

### 环境要求

- 硬件: Atlas 800T A2

### 支持源码编译安装，用户可以执行下述的命令进行包的安装：

```shell
#!/bin/bash
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
pip install -r requirements.txt
```

设置环境变量

```shell
#!/bin/bash
export ASCEND_CUSTOM_PATH=/path/cann/ascend-toolkit
export ASCEND_HOME_PATH=$ASCEND_CUSTOM_PATH

#导入CANN基本环境变量
source $ASCEND_CUSTOM_PATH/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_PATH/latest/lib64:$ASCEND_CUSTOM_PATH/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$LD_LIBRARY_PATH

#配置整网ND消除格式转换算子
export MS_ENABLE_FORMAT_MODE=1

#REF模式和CELL共享
export MS_DISABLE_REF_MODE=1

#内存优化：配置atomic内存单独清零
export MS_GE_ATOMIC_CLEAN_POLICY=1

#内存优化：配置内存扩展模式（实现纯静态图之间内存复用）
export GE_USE_STATIC_MEMORY=2
```

**注：** `ASCEND_CUSTOM_PATH`的`path`替换为CANN包真实地址

### 生成RANK_TABLE_FILE

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件

```shell
#!/bin/bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注：若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成**

RANK_TABLE_FILE 单机8卡参考样例:

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

### 模型权重下载与转换(mindformers权重或huggingface权重选择使用即可)

#### mindformers权重直接使用

本仓库提供已经转换完成的预训练权重用于训练/微调/推理，用户可自行从下方链接拉取后直接使用，Base用于微调。

```shell
#!/bin/bash
mkdir -p ckpt/rank_0
cd ./ckpt/rank_0
wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/wizardcode/wizardcoder_15B.ckpt
cd ../..
```

#### huggingface权重转换后使用

从huggingface下载预训练权重后根据以下步骤进行权重转换，需要下载整个工程，huffingface权重的链接如下：

```shell
#!/bin/bash
mkdir -p ckpt/rank_0
cd ./ckpt
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/added_tokens.json
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/config.json
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/generation_config.json
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/merges.txt
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/pytorch_model.bin
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/special_tokens_map.json
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/tokenizer.json
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/tokenizer_config.json
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/vocab.json
cd ..
```

**注**: 请安装torch=1.11.0和transformers=4.30.2版本

```shell
#!/bin/bash
pip install torch==1.11.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.30.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

下载完成后，运行`/research/wizardcoder/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
#!/bin/bash
python ./research/wizardcoder/convert_weight.py --torch_path ./ckpt --mindspore_path ./ckpt/rank_0
```

```text
# 参数说明
torch_path: huggingface权重保存目录路径下任意权重bin文件，根据文件路径读取目录下全部权重
mindspore_path: mindspore格式的权重保存文件名，如'saved_dir/wizardcoder.ckpt'
```

## WizardCoder-15B

### 训练和微调性能

| config                                                       | task                  | Datasets  | SeqLength | metric | phase             | score     | performance(tokens/s/p)  |
| ------------------------------------------------------------ | --------------------- | --------- | --------- | ------ | ----------------- | --------- | ------------ |
| [wizardcoder_15b](./run_wizardcoder.yaml)    | text_generation   | alpaca      | 2048      | -      | [train](#预训练)  | -         | 798.7  |
| [wizardcoder_15b](./run_wizardcoder.yaml)    | text_generation   | alpaca      | 2048      | -      | [finetune](#微调)  | -         | 798.7  |

```shell
#!/bin/bash
pip install mindspore==2.2.0
pip install mindpet==1.0.2
```

**注：** mindspore-lite==2.2.0需要离线安装，步骤如下：

- 首先下载whl安装包

```shell
#!/bin/bash
mkdir -p mslite_path
cd ./mslite_path
wget https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/lite/release/linux/aarch64/cloud_fusion/python37/mindspore_lite-2.2.0-cp37-cp37m-linux_aarch64.whl
cd ..
```

- 再安装下载好的whl包

```shell
#!/bin/bash
cd ./mslite_path
pip install mindspore_lite-2.2.0-cp37-cp37m-linux_aarch64.whl
```

### 预训练

#### 数据集准备-预训练数据集

当前提供Alpaca数据集的预处理和预训练样例，用于对wizardcoder-15B模型进行预训练。数据集的官方下载链接如下：

```shell
#!/bin/bash
mkdir dataset
cd ./dataset
wget https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
cd ..
```

修改`research/wizardcoder/wizardcoder_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```python
# 原代码 research/wizardcoder/wizardcoder_preprocess.py, 138行
# tokenize_qa()的入参if_jsonl需要设置为False
def tokenize_qa(tokenizer, file_path, max_length, if_jsonl=False):
    ...
```

```shell
#!/bin/bash
python research/wizardcoder/wizardcoder_preprocess.py \
--input_glob ./dataset/alpaca_data.json \
--vocab_file ./ckpt/vocab.json \
--merge_file ./ckpt/merges.txt \
--output_file ./dataset/alpaca_data.mindrecord \
--seq_length 2048
```

#### 预训练启动

- step 1. 修改`research/wizardcoder/run_wizardcoder.yaml`中相关配置

```text
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: './ckpt'          # 添加预训练权重路径
auto_trans_ckpt: True
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'train'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "{path}/adgen.mindrecord"   # 修改训练数据集路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
```

- step 2. 启动微调任务，按照以下步骤启动：

-[x] 1: 根据服务器节点数等信息，修改相应的配置。

```shell
#!/bin/bash
# 以wizardcoder模型为例，默认配置单机8卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：./research/wizardcoder/run_wizardcoder.yaml
parallel_config:
  data_parallel: 1
  model_parallel: 4
  pipeline_stage: 2
  optimizer_shard: True
  micro_batch_num: 8
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

-[x] 2: 执行运行脚本。

```shell
#!/bin/bash
bash research/run_singlenode.sh \
"python research/wizardcoder/run_wizardcoder.py \
--config research/wizardcoder/run_wizardcoder.yaml \
--load_checkpoint ./ckpt \
--use_parallel True \
--run_mode train \
--train_data ./dataset/alpaca_data.mindrecord" \
path/to/rank_table_file [0,8] 8
```

```text
# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径
run_mode: 运行模式，训练时设置为train
train_data: 训练数据集路径
path/to/rank_table_file: 生成RANK_TABLE_FILE步骤中生成的hccl_***_json文件路径
```

### 微调

#### 数据集准备-SFT微调数据集

当前提供codealpaca数据集的预处理和微调样例，用于对wizardcoder-15B模型进行微调。数据集下载链接如下：

```shell
#!/bin/bash
mkdir finetune_dataset
cd ./finetune_dataset
wget https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json
cd ..
```

执行`research/wizardcoder/wizardcoder_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```python
# 原代码 research/wizardcoder/wizardcoder_preprocess.py, 138行
# tokenize_qa()的入参if_jsonl需要设置为False
def tokenize_qa(tokenizer, file_path, max_length, if_jsonl=False):
    ...
```

```bash
# 脚本路径：research/wizardcoder/wizardcoder_preprocess.py
python research/wizardcoder/wizardcoder_preprocess.py \
--input_glob ./finetune_dataset/code_alpaca_20k.json \
--vocab_file ./ckpt/vocab.json \
--merge_file ./ckpt/merges.txt \
--output_file ./finetune_dataset/code_alpaca.mindrecord \
--seq_length 2048
```

#### 全参微调

全参微调需要多卡启动，以`CodeAlpaca-20k`数据集为例,给出了默认配置文件`research/wizardcoder/run_wizardcoder.yaml`。

- step 1. 修改`research/wizardcoder/run_wizardcoder.yaml`中相关配置

```text
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: './output/transformed_checkpoint/'          # 添加预训练权重路径
auto_trans_ckpt: False
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "./finetune_dataset/code_alpaca.mindrecord"   # 修改训练数据集路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# 指令微调时（如code_alpaca数据集），input_columns: ["input_ids", "labels"]
```

- step 2. 启动微调任务，按照以下步骤启动：

-[x] 1: 根据服务器节点数等信息，修改相应的配置。

```shell
# 以wizardcoder模型为例，默认配置单机8卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：./research/wizardcoder/run_wizardcoder.yaml
parallel_config:
  data_parallel: 1
  model_parallel: 4
  pipeline_stage: 2
  optimizer_shard: True
  micro_batch_num: 8
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

-[x] 2: 执行运行脚本。

```shell
bash research/run_singlenode.sh \
"python research/wizardcoder/run_wizardcoder.py \
--config research/wizardcoder/run_wizardcoder.yaml \
--load_checkpoint ./output/transformed_checkpoint/ \
--use_parallel True \
--run_mode finetune \
--train_data ./finetune_dataset/code_alpaca.mindrecord" \
path/to/rank_table_file [0,8] 8
```

```text
# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集路径
path/to/rank_table_file: 生成RANK_TABLE_FILE步骤中生成的hccl_***_json文件路径
```

### 快速推理

**注：** 推理部分需要更新如下环境变量

```shell
#!/bin/bash
unset MS_DISABLE_REF_MODE=1
export MS_ENABLE_REF_MODE=1
```

#### 基于高阶接口的推理

- step 1. 配置文件设置，添加tokenizer路径`vocab_file`和`merge_file`，并设置`batch_size`值为`1`

在使用Trainer接口进行推理时，若用户自行下载wizardcoder权重，请在启动前先在配置文件中将vocab.json和merges.txt的路径自行配置，配置项为vocab_file和merge_file。

```yaml
# research/wizardcoder/run_wizardcoder.yaml
# runner config
runner_config:
  epochs: 1
  batch_size: 1                 # batch_size设为1
  sink_mode: True
  sink_size: 2
...
processor:
 return_tensors: ms
 tokenizer:
   unk_token: '<|endoftext|>'
   bos_token: '<|endoftext|>'
   eos_token: '<|endoftext|>'
   pad_token: '[PAD]'
   vocab_file: 'vocab.json'        # 添加vocab.json的路径
   merge_file: 'merges.txt'        # 添加merges.txt的路径
   type: WizardCoderTokenizer
```

相关文件的下载链接如下：[vocab.json](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/wizardcode/vocab.json) 和 [merges.txt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/wizardcode/merges.txt)

- step 2. Trainer接口启动推理

wizardcoder的高阶接口使用脚本已集成在run_wizardcoder.py脚本中，运行此脚本命令示例：

其中`ckpt_path_or_dir`为模型文件地址，如：{path}/wizardcoder.ckpt

```shell
python run_wizardcoder.py \
--config "run_wizardcoder.yaml" \
--run_mode predict \
--use_parallel False \
--load_checkpoint ckpt_path_or_dir \
--predict_data '使用python编写快速排序代码' \
--device_id 0

# output: 快速排序（QuickSort）是一种非常高效的排序算法，它是选择排序算法的一个非常有效的改进版本。它的基本思想是通过一趟排序将待排记录分隔成独立的两部分，其中一部分记录的元素值比另一部分的元素值小，然后再按此方法对子部分继续进行排序，直到整个序列有序。\n\nPython中的快速排序算法可以实现如下：\n\n```\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    else:\n        pivot = arr[0]\n        left = [x for x in arr[1:] if x <= pivot]\n        middle = pivot\n        right = [x for x in arr[1:] if x > pivot]\n        return quicksort(left) + [middle] + quicksort(right)\n```\n\n这个函数接收一个列表作为输入，并返回一个排序后的列表。\n\n该函数首先检查输入列表的长度，如果长度为0或1，直接返回列表。否则，选取第一项作为分区点（pivot），然后将列表中所有小于等于这个分区点的元素放入左子列表，大于分区点的元素放入右子列表。最后，递归地调用左子列表和右子列表的排序函数。\n\n这样，当递归到最底层的时候，每个子列表中只包含一个元素，这时候就不用再递归了。最后，将两个子列表连接起来，并加上分区点，得到一个排序后的列表。
```

#### Pipeline推理(单卡)

在使用Pipeline接口进行推理时，用户自行下载Wizardcoder-15B权重和tokenizer文件，在启动前自行配置路径
WizardCoderConfig的入参use_past=False为自回归推理，use_past=True为增量推理

**注：** 使用如下脚本推理，其中`wizardcoder_model_path`是权重存放的地址，`tokenizer_path`是存放vocab.json和merges.txt的目录地址

```text
import os
import sys

sys.path.append(os.path.abspath("../.."))
sys.path.insert(0, os.getcwd().split('research')[0])
from mindspore import context
from mindformers.pipeline import pipeline

from wizardcoder_config import WizardCoderConfig
from wizardcoder import WizardCoderLMHeadModel
from wizardcoder_tokenizer import WizardCoderTokenizer

context.set_context(device_id=0, mode=0)

# init model

wizardcoder_model_path = "/path/wizardcoder_15b.ckpt" # 添加模型文件地址
wizardcoder_config = WizardCoderConfig(
    batch_size=1,
    seq_length=2048,
    n_position=8192,
    vocab_size=49153,
    hidden_size=6144,
    num_layers=40,
    num_heads=48,
    eos_token_id=0,
    pad_token_id=49152,
    checkpoint_name_or_path=wizardcoder_model_path,
    use_past=True # False为自回归推理，True为增量推理
)
wizardcoder_model = WizardCoderLMHeadModel(config=wizardcoder_config)
wizardcoder_model.add_flags_recursive(is_first_iteration=True)

# init tokenizer

tokenizer_path = "/path/Wizardcoder-15B/tokenizer_path/" # Wizardcoder-15B tokenizer path
tokenizer = WizardCoderTokenizer(
    vocab_file=tokenizer_path + "vocab.json",    # tokenizer_path为存放vocab.json和merges.txt的地址
    merge_file=tokenizer_path + "merges.txt"
)
pipeline_task = pipeline(task="text_generation", model=wizardcoder_model, tokenizer=tokenizer)
input_data = "使用python编写快速排序代码"
pipeline_result = pipeline_task([input_data],
                                do_sample=False,
                                max_length=2048)
print(pipeline_result)


# output: [{'text_generation_text': ['使用python编写快速排序代码，并分析其时间复杂度。\r\n\r\n快速排序是一种分治算法，它的基本思想是：通过一趟排序将待排记录分隔成独立的两部分，其中一部分记录的关键字均比另一部分的关键字小，则可分别对这两部分记录继续进行排序，以达到整个序列有序。\r\n\r\n快速排序的步骤如下：\r\n\r\n1. 从数列中挑出一个元素，称为 “基准”（pivot）\r\n2. 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作。\r\n3. 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。\r\n\r\n快速排序的时间复杂度为O(nlogn)，最坏情况下的时间复杂度为O(n^2)，平均情况下的时间复杂度为O(nlogn)。\r\n\r\n下面是Python代码实现的快速排序：\r\n\r\n```python\r\ndef quick_sort(arr):\r\n    if len(arr) <= 1:\r\n        return arr\r\n    else:\r\n        pivot = arr[0]\r\n        left = []\r\n        right = []\r\n        for i in range(1, len(arr)):\r\n            if arr[i] < pivot:\r\n                left.append(arr[i])\r\n            else:\r\n                right.append(arr[i])\r\n        return quick_sort(left) + [pivot] + quick_sort(right)\r\n```\r\n\r\n该代码的基本思路是：\r\n\r\n1. 如果数组的长度小于等于1，则直接返回数组。\r\n2. 选择数组的第一个元素作为基准值。\r\n3. 遍历数组，将比基准值小的元素放到左边，将比基准值大的元素放到右边。\r\n4. 递归地对左边和右边的子数组进行排序。\r\n5. 将左边子数组、基准值、右边子数组合并成一个新的数组。\r\n\r\n下面是该代码的时间复杂度分析：\r\n\r\n- 最坏情况下的时间复杂度：当数组的长度为n，且每次选择的基准值都为数组的第一个元素时，每次递归都需要进行n-1次，因此最坏情况下的时间复杂度为O(n^2)。\r\n- 平均情况下的时间复杂度：每次选择的基准值都为数组的中间元素，每次递归都需要进行logn次，因此平均情况下的时间复杂度为O(nlogn)。\r\n- 最优情况下的时间复杂度：当数组的长度为n，且每次选择的基准值都为数组的第一个元素时，每次递归都需要进行logn次，因此最优情况下的时间复杂度为O(nlogn)。']}]

```

#### Pipeline推理(单机多卡)

以单机4卡分布式推理为例，设置dp=1, mp=4

- step 1. yaml配置

修改run_wizardcoder.yaml中的配置项，只需要修改如下yaml中的vocab_file和merge_file地址

```yaml
use_parallel: True   # 单机多卡或多机多卡必须设为True
parallel_config:
  data_parallel: 1
  model_parallel: 4
  pipeline_stage: 1
model:
  model_config:
    use_seq_parallel: False
    use_past: True   # False为自回归推理，True为增量推理
    checkpoint_name_or_path: ""
processor:
  tokenizer:
    vocab_file: '/tokenizer_path/vocab.json'
    merge_file: '/tokenizer_path/merges.txt'
    type: WizardCoderTokenizer
  type: WizardCoderProcessor
...
```

- step 2. 切分权重

```text
    └── distribute_model_ckpt_path
        └── rank_0
            └── checkpoint_0.ckpt
        └── rank_1
            └── checkpoint_1.ckpt
        └── rank_2
            └── checkpoint_2.ckpt
        └── rank_3
            └── checkpoint_3.ckpt
```

- step 3. 配置单机4卡环境

运行`mindformers/tools/hccl_tools.py`生成`RANK_TABLE_FILE`的json文件hccl_4p_0123_127.0.0.1.json

```text
python mindformers/tools/hccl_tools.py --device_num "[0,4)"
```

- step 4. 推理脚本

``` python
# test_wizardcoder_pipeline_dist.py
import os
import sys
import argparse
import numpy as np
sys.path.append(os.path.abspath("../.."))
sys.path.insert(0, os.getcwd().split('research')[0])
import mindspore as ms
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net

from mindformers import AutoConfig, AutoTokenizer, AutoModel, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig, TransformerOpParallelConfig
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools.utils import str2bool, get_real_rank
from mindformers import Trainer, MindFormerConfig, MindFormerRegister, MindFormerModuleType

from wizardcoder_config import WizardCoderConfig
from wizardcoder_tokenizer import WizardCoderTokenizer
from wizardcoder import WizardCoderLMHeadModel


def context_init(use_parallel=False, device_id=0):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=device_id)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                full_batch=True)
    init_context(use_parallel=use_parallel,
                 context_config=context_config,
                 parallel_config=parallel_config)


def main(model_type='wizardcoder',
         config_path="run_wizardcoder.yaml",
         use_parallel=False,
         device_id=0,
         checkpoint_path=""):
    """main function."""
    # 初始化单卡/多卡环境
    context_init(use_parallel, device_id)
    inputs = ["使用python编写快速排序代码"] * 2
    config = MindFormerConfig(os.path.realpath(config_path))
    # set model config
    model_config = WizardCoderConfig.from_pretrained(os.path.realpath(config_path))
    model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    if checkpoint_path and not use_parallel:
        model_config.checkpoint_name_or_path = checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = WizardCoderTokenizer(config.processor.tokenizer.vocab_file,
                                     config.processor.tokenizer.merge_file)
    # build model from config
    network = WizardCoderLMHeadModel(model_config)
    network.add_flags_recursive(is_first_iteration=True)

    # if use parallel, load distributed checkpoints
    if use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(get_real_rank()), "checkpoint_{}.ckpt".format(get_real_rank()))
        print("ckpt path: %s", str(ckpt_path))

        # shard model and load sharded ckpt
        model = Model(network)
        model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(network, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    text_generation_pipeline = pipeline(task="text_generation", model=network, tokenizer=tokenizer)
    outputs = text_generation_pipeline(inputs, do_sample=False, max_length=2048)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='wizardcoder', type=str,
                        help='which model to use.')
    parser.add_argument('--config_path', default='run_wizardcoder.yaml', type=str,
                        help='config path')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='whether use parallel.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    args = parser.parse_args()

    main(args.model_type,
         args.config_path,
         args.use_parallel,
         args.device_id,
         args.checkpoint_path)
```

使用如下bash启动脚本来运行pipeline分布式推理

```bash
# pipeline_dist.sh

CHECKPOINT_PATH=$2
export RANK_TABLE_FILE=$1

# define variable
export RANK_SIZE=4
export START_RANK=0 # this server start rank
export END_RANK=4 # this server end rank

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export RANK_ID=$i
    export DEVICE_ID=$((i-START_RANK))
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./test_wizardcoder_pipeline_dist.py --use_parallel True --checkpoint_path $CHECKPOINT_PATH &> mindformers_$RANK_ID.log &
done

```

执行命令bash pipeline_dist.sh hccl_4p_0123_127.0.0.1.json distribute_model_ckpt_path/

推理结果

```text
{'text_generation_text': ['使用python编写快速排序代码，并分析其时间复杂度。\r\n\r\n快速排序是一种分治算法，它的基本思想是：通过一趟排序将待排记录分隔成独立的两部分，其中一部分记录的关键字均比另一部分的关键字小，则可分别对这两部分记录继续进行排序，以达到整个序列有序。\r\n\r\n快速排序的步骤如下：\r\n\r\n1. 从数列中挑出一个元素，称为 “基准”（pivot）\r\n2. 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作。\r\n3. 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。\r\n\r\n快速排序的时间复杂度为O(nlogn)，最坏情况下的时间复杂度为O(n^2)，平均情况下的时间复杂度为O(nlogn)。\r\n\r\n下面是Python代码实现的快速排序：\r\n\r\n```python\r\ndef quick_sort(arr):\r\n    if len(arr) <= 1:\r\n        return arr\r\n    else:\r\n        pivot = arr[0]\r\n        left = []\r\n        right = []\r\n        for i in range(1, len(arr)):\r\n            if arr[i] < pivot:\r\n                left.append(arr[i])\r\n            else:\r\n                right.append(arr[i])\r\n        return quick_sort(left) + [pivot] + quick_sort(right)\r\n```\r\n\r\n该代码的基本思路是：\r\n\r\n1. 如果数组的长度小于等于1，则直接返回数组。\r\n2. 选择数组的第一个元素作为基准值。\r\n3. 遍历数组，将比基准值小的元素放到左边，将比基准值大的元素放到右边。\r\n4. 递归地对左边和右边的子数组进行排序。\r\n5. 将左边子数组、基准值、右边子数组合并成一个新的数组。\r\n\r\n下面是该代码的时间复杂度分析：\r\n\r\n- 最坏情况下的时间复杂度：当数组的长度为n，且每次选择的基准值都为数组的第一个元素时，每次递归都需要进行n-1次，因此最坏情况下的时间复杂度为O(n^2)。\r\n- 平均情况下的时间复杂度：每次选择的基准值都为数组的中间元素，每次递归都需要进行logn次，因此平均情况下的时间复杂度为O(nlogn)。\r\n- 最优情况下的时间复杂度：当数组的长度为n，且每次选择的基准值都为数组的第一个元素时，每次递归都需要进行logn次，因此最优情况下的时间复杂度为O(nlogn)。']}
```

### mindspore lite推理

- step 1. yaml配置

修改run_wizardcoder.yaml中的配置项，将模型权重文件的路径名写入`checkpoint_name_or_path`

```yaml
use_parallel: False
model:
  model_config:
    use_seq_parallel: False
    use_past: True # False为自回归推理，True为增量推理
    checkpoint_name_or_path: "/path/wizardcoder_15b.ckpt"
...
```

- step 2. 生成mindir模型文件

基于配置文件run_wizardcoder.yaml生成自回归推理的mindir文件`wizardcoder-15b_mslite_autoregressive/prefill_2k_bs1_graph.mindir`

```text
# export_wizardcoder_autoregressive.py
import os
import sys

sys.path.append(os.path.abspath("../.."))
sys.path.insert(0, os.getcwd().split('research')[0])

import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype

from wizardcoder_config import WizardCoderConfig
from wizardcoder import WizardCoderLMHeadModel

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
BS = 1
SEQ_LENGTH = 2048
config_path = 'run_wizardcoder.yaml'
config = WizardCoderConfig.from_pretrained(os.path.realpath(config_path))
config.use_past = False    # False为自回归推理，True为增量推理
model = WizardCoderLMHeadModel(config)
model.set_train(False)

model.add_flags_recursive(is_first_iteration=True)
input_ids = ms.Tensor(np.ones((BS, SEQ_LENGTH)), mstype.int32)
ms.export(model,
    input_ids,
    file_name=f"wizardcoder-15b_mslite_autoregressive/prefill_2k_bs{BS}",
    file_format="MINDIR")
```

基于配置文件run_wizardcoder.yaml生成增量推理的mindir文件`wizardcoder-15b_mslite_inc/prefill_2k_bs1_graph.mindir`和`wizardcoder-15b_mslite_inc/decode_2k_bs1_graph.mindir`

```text
# export_wizardcoder_inc.py
import os
import sys
sys.path.append(os.path.abspath("../.."))
sys.path.insert(0, os.getcwd().split('research')[0])
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype

from wizardcoder_config import WizardCoderConfig
from wizardcoder import WizardCoderLMHeadModel

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
BS = 1
SEQ_LENGTH = 2048

# wizardcoder配置文件

config_path = 'run_wizardcoder.yaml'
config = WizardCoderConfig.from_pretrained(os.path.realpath(config_path))
config.use_past = True # False为自回归推理，True为增量推理

model = WizardCoderLMHeadModel(config)
model.set_train(False)

# 全量推理 prefill

model.add_flags_recursive(is_first_iteration=True)
input_ids = ms.Tensor(np.ones((BS, SEQ_LENGTH)), mstype.int32)
input_position = ms.Tensor([127]*BS, mstype.int32)
init_reset = ms.Tensor([False], mstype.bool_)
batch_valid_length = ms.Tensor([[128]*BS], mstype.int32)
ms.export(model, input_ids, None, None, input_position, init_reset, batch_valid_length, file_name=f"wizardcoder-15b_mslite_inc/prefill_2k_bs{BS}", file_format="MINDIR")

# 增量推理 decode

model.add_flags_recursive(is_first_iteration=False)
input_ids = ms.Tensor(np.ones((BS, 1)), mstype.int32)
input_position = ms.Tensor([128]*BS, mstype.int32)
init_reset = ms.Tensor([True], mstype.bool_)
batch_valid_length = ms.Tensor([[129]*BS], mstype.int32)
ms.export(model, input_ids, None, None, input_position, init_reset, batch_valid_length, file_name=f"wizardcoder-15b_mslite_inc/decode_2k_bs{BS}", file_format="MINDIR")
```

- step 3. GE配置

GE配置文件context.cfg

```text
[ascend_context]
plugin_custom_ops=All
provider=ge
[ge_session_options]
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
```

- step 4. mslite推理

指定使用tokenizer配置，配置步骤2生成的mindir文件，配置步骤3中的GE文件context.cfg

```text
# inference_wizardcoder_mslite.py
import sys
import os

sys.path.append(os.path.abspath("../.."))
sys.path.insert(0, os.getcwd().split('research')[0])

from mindspore import context
from mindformers.pipeline import pipeline

from wizardcoder_config import WizardCoderConfig
from wizardcoder_tokenizer import WizardCoderTokenizer


context.set_context(device_id=0, mode=0)
tokenizer_path = "/tokenizer_path/" # tokenizer_path中存放有vocab.json和merges.txt
tokenizer = WizardCoderTokenizer(
    vocab_file=tokenizer_path + "vocab.json",
    merge_file=tokenizer_path + "merges.txt"
)
use_past = True # False为自回归推理，True为增量推理
if use_past:
    model_path = ("wizardcoder-15b_mslite_inc/prefill_2k_bs1_graph.mindir", "wizardcoder-15b_mslite_inc/decode_2k_bs1_graph.mindir")
else:
    model_path = ("wizardcoder-15b_mslite_autoregressive/prefill_2k_bs1_graph.mindir", None)
ge_config_path = "context.cfg"
pipeline_task = pipeline(task="text_generation", model=model_path, backend="mslite", tokenizer=tokenizer, ge_config_path=ge_config_path, model_type="mindir", infer_seq_length=2048, add_special_tokens=False)

input_data = ["使用python编写快速排序代码"]

pipeline_result = pipeline_task(input_data, do_sample=False, max_length=2048,  eos_token_id=0, pad_token_id=49152, skip_special_tokens=True)

print(pipeline_result[0])

# ['使用python编写快速排序代码，并分析其时间复杂度。\r\n\r\n快速排序是一种分治算法，它的基本思想是：通过一趟排序将待排记录分隔成独立的两部分，其中一部分记录的关键字均比另一部分的关键字小，则可分别对这两部分记录继续进行排序，以达到整个序列有序。\r\n\r\n快速排序的步骤如下：\r\n\r\n1. 从数列中挑出一个元素，称为 “基准”（pivot）\r\n2. 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作。\r\n3. 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。\r\n\r\n快速排序的时间复杂度为O(nlogn)，最坏情况下的时间复杂度为O(n^2)，平均情况下的时间复杂度为O(nlogn)。\r\n\r\n下面是Python代码实现的快速排序：\r\n\r\n```python\r\ndef quick_sort(arr):\r\n    if len(arr) <= 1:\r\n        return arr\r\n    else:\r\n        pivot = arr[0]\r\n        left = []\r\n        right = []\r\n        for i in range(1, len(arr)):\r\n            if arr[i] < pivot:\r\n                left.append(arr[i])\r\n            else:\r\n                right.append(arr[i])\r\n        return quick_sort(left) + [pivot] + quick_sort(right)\r\n```\r\n\r\n该代码的基本思路是：\r\n\r\n1. 如果数组的长度小于等于1，则直接返回数组。\r\n2. 选择数组的第一个元素作为基准值。\r\n3. 遍历数组，将比基准值小的元素放到左边，将比基准值大的元素放到右边。\r\n4. 递归地对左边和右边的子数组进行排序。\r\n5. 将左边子数组、基准值、右边子数组合并成一个新的数组。\r\n\r\n下面是该代码的时间复杂度分析：\r\n\r\n- 最坏情况下的时间复杂度：当数组的长度为n，且每次选择的基准值都为数组的第一个元素时，每次递归都需要进行n-1次，因此最坏情况下的时间复杂度为O(n^2)。\r\n- 平均情况下的时间复杂度：每次选择的基准值都为数组的中间元素，每次递归都需要进行logn次，因此平均情况下的时间复杂度为O(nlogn)。\r\n- 最优情况下的时间复杂度：当数组的长度为n，且每次选择的基准值都为数组的第一个元素时，每次递归都需要进行logn次，因此最优情况下的时间复杂度为O(nlogn)。']
```

### 推理性能评测

#### 评测结果

| batch size | seq_length | Atlas 800T A2(400T) tokens/s | A800tokens/s | 对比          |
|------------|------------|----------------------|--------------|-------------|
| 32         | 32         | 407.4103739          | 108.1633254  | 3.76662212  |
| 32         | 64         | 404.1433301          | 156.4472315  | 2.583256516 |
| 32         | 128        | 406.220563           | 260.4452566  | 1.559715728 |
| 32         | 256        | 403.642159           | 329.6634958  | 1.224406597 |
| 32         | 512        | 388.4709184          | 330.4193671  | 1.175690523 |
| 32         | 1024       | 336.0791015          | 345.8010141  | 0.971885818 |
| 32         | 2048       | 251.154984           | 350.2757495  | 0.717020759 |
| 平均         | -          | 371.017347           | 268.7450629  | 1.380555026 |

#### 评测流程

- **step 1: Atlas 800T A2 + mindspore lite推理**

-[x] 基于配置文件run_wizardcoder.yaml生成增量推理的mindir文件`wizardcoder-15b_mslite_inc/prefill_2k_bs1_graph.mindir`和`wizardcoder-15b_mslite_inc/decode_2k_bs1_graph.mindir`

```bash
python export_wizardcoder_inc.py
--device_id DEVICE_ID
--batch_size BATCH_SIZE
--seq_length SEQ_LENGTH
--model_path path/wizardcoder.ckpt
```

其中，path文件夹中存放有`wizardcoder.ckpt`模型权重

-[x] 执行性能测试脚本，生成性能结果

```bash
python inference_wizardcoder_mslite.py
--device_id DEVICE_ID
--batch_size BATCH_SIZE
--seq_length SEQ_LENGTH
--tokenizer_path tokenizer_path/
```

其中，tokenizer_path文件夹中存放有`vocab.json`和`merges.txt`文件

- **step 2: A800 + Pytorch推理**

- [x] 执行性能测试脚本，生成性能结果

**注**: 请安装torch=1.11.0和transformers=4.30.2版本

```bash
CUDA_VISIBLE_DEVICES=DEVICE_ID python inference_wizardcoder_pytorch.py
--base_model BASE_MODEL
--batch_size BATCH_SIZE
--seq_length SEQ_LENGTH
```

DEVICE_ID设为使用gpu的id号

base_model文件夹中存放huggingface wizardcoder-15B的模型文件，具体参见[wizardcoder-15B](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/tree/main)，将所有文件下载并保存到BASE_MODEL路径下即可

### 开源数据集评测

#### 评测结果

**注：** 评测结果基于开源的预训练模型

|                                | MBPP Pass@1 |
|--------------------------------|-------------|
| Atlas 800T A2 + Mindspore (在线推理)        | 50.8        |
| Atlas 800T A2 + Mindspore (离线推理)        | 50.8        |
| A100 + Pytorch                 | 50.6        |

#### MBPP评测流程

- step 1: 安装Code Generation LM Evaluation Harness

```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
pip install -e .
```

- step 2: 生成推理结果

```bash
output_path=test/mbpp          # 离线推理结果保存路径
# output_path=test/mbpp_online # 在线推理结果保存路径
mbpp_path=mbpp.test.jsonl
tokenizer_path=""              # tokenizer的vocab.json和merges.txt所在目录
model_path="wizardcoder.ckpt"  # wizardcoder模型文件
mkdir -p ${output_path}
echo 'Output path: '$output_path

# 500条测试数据，如果使用8卡，每张卡分配63条数据
npu_num=4
step=130
for ((i = 0; i < $npu_num; i++)); do
  start_index=$((i * step))
  end_index=$(((i + 1) * step))
  npu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on NPU' ${npu}
  # 离线推理
  python mbpp_gen_mslite.py --start_index ${start_index} --end_index ${end_index} --output_path ${output_path} --mbpp_path ${mbpp_path} --device_id ${npu} --tokenizer_path ${tokenizer_path} --model_path ${model_path} &> mbpp_$npu.log &
  # 在线推理
  # python mbpp_gen_online.py --start_index ${start_index} --end_index ${end_index} --output_path ${output_path} --mbpp_path ${mbpp_path} --device_id ${npu} --tokenizer_path ${tokenizer_path} --model_path ${model_path} &> mbpp_online_$npu.log &
done
```

500条测试样本的推理结果保存在`test/mbpp`目录下

- step 3: 汇总推理结果，提取推理结果中的可行性代码

```bash
# 离线推理结果合并
python mbpp_process.py --path test/mbpp --out_path mbpp_npu.json
# 在线推理结果合并
python mbpp_process.py --path test/mbpp_online --out_path mbpp_npu_online.json
```

汇总后的离线推理结果保存在`mbpp_npu.json`中，汇总后的在线推理结果保存在`mbpp_npu_online.json`

- step 4: 生成测试分数

进入`Code Generation LM Evaluation Harness`安装目录，将汇总后的推理结果文件`mbpp_npu.json`和`mbpp_npu_online.json`复制到当前文件夹，执行如下命令生成推理分数

```bash
# 离线推理结果评测
python  main.py   --tasks mbpp  --allow_code_execution  --load_generations_path mbpp_npu.json
# 在线推理结果评测
python  main.py   --tasks mbpp  --allow_code_execution  --load_generations_path mbpp_npu_online.json
```
