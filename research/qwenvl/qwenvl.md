# Qwen-VL

## 模型描述

Qwen-VL 是阿里云研发的大规模视觉语言模型（Large Vision Language Model, LVLM）。Qwen-VL 可以以图像、文本、检测框作为输入，并以文本和检测框作为输出。

```text
@article{Qwen-VL,
  title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}
```

## 仓库介绍

`Qwen-VL` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   qwenvl
     ├── qwenvl_config.py         # 配置文件
     ├── qwenvl_tokenizer.py      # tokenizer
     └── qwenvl.py                # 模型实现
   ```

2. 模型配置：

   ```text
   qwenvl
     ├── predict_qwenvl_9.6b.yaml            # qwenvl推理启动配置
     ├── finetune_qwenvl_9.6b.yaml           # qwenvl微调启动配置（2k）
     └── finetune_qwenvl_9.6b_bf16.yaml      # qwenvl微调启动配置（2k，bf16）
   ```

3. 环境准备和任务启动脚本：

   ```text
   qwenvl
     ├── qwenvl_dataset.py        # 实际传入训练的数据集，实现了batch，transform等操作操作
     ├── qwenvl_dataloader.py     # 加载json格式的qwenvl数据
     ├── qwenvl_transform.py      # 数据加载时使用的数据转换
     ├── qwenvl_processor.py      # 推理时候使用的数据处理
     ├── convert_weight.py        # 权重转换脚本
     ├── data_convert.py          # 数据预处理转换脚本
     └── run_qwenvl.py            # QwenVL高阶接口脚本
   ```

## 前期准备

### [mindformers安装](../../README.md#二mindformers安装)

### 环境搭建

- 硬件：Atlas 800T A2
- MindSpore：2.3.0
- MindFormers版本：1.2.0
- Python：3.8+

### 模型权重准备

本仓库提供已经转换完成的预训练权重、词表文件用于微调、推理，用户可自行从下方链接拉取后使用。

- [Qwen-VL-Base](https://openmind.cn/coderepo/web/v1/file/MindSpore-Lab/Qwen-VL/main/media/qwenvl_base_fp16.ckpt)
- [qwen.tiktoken](https://openmind.cn/coderepo/web/v1/file/MindSpore-Lab/Qwen-VL/main/media/qwen.tiktoken)

#### 从huggingface版本权重文件转换

也可选择从huggingface下载预训练权重后根据以下步骤进行权重转换，需要下载整个工程。huggingface权重的下载链接如下：

- [Qwen-VL-Base](https://huggingface.co/Qwen/Qwen-VL/tree/main)

1. 安装权重转换必须软件包

```shell
pip install torch
pip install transformers  # 如果transformers使用tokenizers版本不是0.15.0，在权重转换完成后重装tokenizers版本为0.15.0
pip install einops transformers_stream_generator accelerate
```

2. 运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model qwenvl --input_path /path/to/hf/dir \
--output_path /path/to/qwenvl_ms.ckpt \
--dtype fp16
```

参数说明：

`input_path`：传入从`Huggingface`下载好模型文件夹；
`output_path`：是用于存储转换后权重的路径；
`dtype`：转换权重的精度选择。

## 微调

指令微调阶段即Qwen-VL论文中的第三阶段，在这个阶段中，会将ViT进行冻结，仅训练QwenLM及CrossAttn部分参数，训练参数约7.78B，
使用MindFormers进行微调时在Atlas 800T A2的性能数据如下（image_size=448，seq_length=2048，单机八卡使用LLaVA-instruct数据集）：

| Model            | Global Batch Size | tokens/p/s |
|------------------|-------------------|------------|
| Qwen-VL-9.6B(FA) | 32                | 2519       |

### 数据集准备

目前QwenVL微调数据集格式同QwenVL开源使用数据集格式一致，如下示例：

```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>assets/demo.jpeg</img>\n图中的狗是什么品种？"
      },
      {
        "from": "assistant",
        "value": "图中是一只拉布拉多犬。"
      },
      {
        "from": "user",
        "value": "框出图中的格子衬衫"
      },
      {
        "from": "assistant",
        "value": "<ref>格子衬衫</ref><box>(588,499),(725,789)</box>"
      }
    ]
  }
]
```

#### yaml数据集配置

```yaml
data_loader:
  type: QwenVLDataLoader
  dataset_dir: "/location/of/images"
  annotation_file: "conversation_file.json"
  column_names: [ "image", "text" ]
  shuffle: True
  extra_kwargs:
    max_img_len: 1
  map_function_kwargs:
    user_role_name: user
    assistant_role_name: assistant
text_transforms:
  type: QwenVLTransform
  max_length: 2049
tokenizer:
  type: QwenVLTokenizer
  vocab_file: "/path/to/vocab_file"
```

data_loader的参数解释：

- type: 数据集加载器的类型，固定为QwenVLDataLoader。
- dataset_dir: 图片数据所在文件夹。对话数据中`<img>relative_path_to_img.jpg</img>`
  对应图片路径为`os.path.join(dataset_dir, relative_path_to_img.jpg)`。
- annotation_file: json格式的对话数据路径。
- column_names: 数据集输出的列名。一般为image和text。
- shuffle: 是否打乱数据集。
- extra_kwargs: 额外的参数。
    - max_img_len: 对话中内容中支持的最大图片数量。
    - map_function_kwargs: map_function的参数。
        - user_role_name: 提出问题方的名称，对应数据集中的from。
        - assistant_role_name: 回答问题方的名称，对应数据集中另外一个的from。

text_transforms的参数解释：

- type: 数据集加载器的类型，固定为QwenVLTransform。
- max_length: 语言模型的seq_length+1，当修改语言模型的seq_length时需要同步修改该值。

#### 示例数据集制作

Qwen-VL开源模型中未开源相关数据集，并且当前公开数据集中，没有同Qwen-VL一致的数据集格式，以下提供一个使用公开数据集转换为Qwen-VL微调数据集格式的方式。

1. 在huggingface上下载[LlaVA-Instruct-150K中的detail_23k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/detail_23k.json?download=true);

2. 在[COCO](https://cocodataset.org/#download)上下载2014 Train images数据集。

3. 通过如下命令运行目录中`data_convert.py`脚本

   ```shell
   python data_convert.py --data_path /path/to/detail_23k.json --image_location /location/of/coco/train2014 --output_path /path/to/converted/json --user_role_name user --assistant_role_name assistant
   ```

   其中`--data_path`表示原始对话数据路径，`--image_location`表示COCO
   train2014文件夹所在路径，路径不包含train2014，`--output_path`表示转换后对话数据保存路径, `--user_role_name`
   表示转换后对话中用户名称，`--assistant_role_name`表示转换后对话中助手名称。

#### 启动微调

1. 当前模型已提供运行微调配置`finetune_qwenvl_9.6b.yaml`，可在此配置文件上根据实际运行情况更改配置。 Qwen-VL支持在LLM模型部分使用
   **Flash Attention算法**进行微调，请参考 [Flash Attention使用文档](../../docs/feature_cards/Training_Algorithms.md#flash-attention)。

2. 修改`finetune_qwenvl_9.6b.yaml中相关配置，开启权重自动转换，加载完整权重。

   ```yaml
   load_checkpoint: '/path/model_dir' # 权重路径，也可通过启动命令自动传入
   auto_trans_ckpt: True              # 打开自动权重转换，也可通过启动命令自动传入
   use_parallel: True
   run_mode: 'finetune'

   train_dataset: &train_dataset
     data_loader:
      type: QwenVLDataLoader
      dataset_dir: "/location/of/coco/train2014"     # 根据实际位置进行配置，当使用示例数据集时为train2014文件夹所在路径，配置项不包含train2014
      annotation_file: "/path/to/converted/json"     # 根据实际位置，填写json文件所在路径
      column_names: [ "image", "text" ]
      shuffle: True
      extra_kwargs:
        max_img_len: 1                              # 根据数据集中对话实际包含图片数量进行配置，在使用示例数据集时为1
      map_function_kwargs:
        user_role_name: user                        # 根据实际配置，修改为用户角色名
        assistant_role_name: assistant              # 根据实际配置，修改为助手角色名

      tokenizer:
        type: QwenVLTokenizer
        vocab_file: "/path/to/qwen.tiktoken"         # 根据词表所在位置，填写词表所在路径

   processor:
    tokenizer:
      vocab_file: "/path/to/qwen.tiktoken"          # 根据词表所在位置，填写词表所在路径
    ```

3. 启动微调任务

运行如下命令启动单机8卡微调任务。

```shell
cd research/qwenvl
bash ../../scripts/msrun_launcher.sh "python run_qwenvl.py \
--config finetune_qwenvl_9.6b.yaml \
--run_mode finetune \
--load_checkpoint /path/to/ckpt \
--use_parallel True \
--auto_trans_ckpt True \
--vocab_file /path/to/qwen.tiktoken" 8

# 以上除config外其他传参如果在yaml文件中已经配置，可以在启动命令中不再传入
# 参数说明
# config: 配置文件路径
# run_mode: 运行模式，微调时设置为finetune
# load_checkpoint: 当使用分布式权重时传入权重文件夹路径model_dir，权重按照'model_dir/rank_0/xxx.ckpt'格式存放，传入完整权重时传入ckpt路径
# auto_trans_ckpt: 自动权重转换开关，当传入完整权重时打开
```

## 推理

当前Qwen-VL推理仅支持单样本推理，且不支持增量推理，使用推理时已提供配置文件`predict_qwenvl_9.6b.yaml`，可在此配置文件上根据实际情况进行修改。

### 使用开源权重推理

将`predict_qwenvl_9.6b.yaml`中的`use_past`配置关闭：

```yaml
load_checkpoint: "/path/to/ckpt"      # 权重所在路径，也可通过启动命令指定
model:
  model_config:
    type: QwenVLConfig
    # ......
    use_past: False                   # 关闭增量推理，也可通过启动命令传入
```

使用如下命令启动推理

```shell
cd research/qwenvl

python run_qwenvl.py --config predict_qwenvl_9.6b.yaml \
--use_parallel False \
--load_checkpoint /path/to/ckpt \
--vocab_file /path/to/qwen.tiktoken \
--device_id 0 --run_mode predict \
--image_path /path/to/demo.jpeg \
--image_size 448 \
--prompt "Describe the image in English:"
```

例如使用Qwen-VL Base权重对如下图片进行推理

![`demo.jpeg`](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwenvl/demo.jpeg)
图片链接：https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwenvl/demo.jpeg

生成结果如下（输出结果中，`<img></img>`间为实际传入图片路径）：

```text
Picture 1: <img>/path/to/demo.jpeg</img>
Describe the image in English: A woman and a dog sitting on the beach.<|endoftext|>
```

### 使用微调后权重推理

当使用多卡微调后，保存的权重为分布式权重，需要对分布式权重进行合并。具体过程如下：

- 参照[权重转换文档](../../docs/feature_cards/Transform_Ckpt.html)将`output/checkpoint_network`文件夹下保存的分布式权重合并成完整权重；
  完成以上权重合并过程后，可以参考前文使用开源权重推理进行推理，在执行命令时传入`load_checkpoint`
  值为以`_merge_pos_embedding.ckpt`结尾的ckpt路径。

### BF16 支持

当前版本仅支持 bf16 数据类型的训练，暂不支持推理。

- `convert_weight.py` 脚本默认的数据类型已经改为与原始权重一致（对于通义千问而言，即`bfloat16`）;
- 推理时可将YAML配置中的`compute_dtype`和`param_init_type`改为`float16`;
- 如果打算基于 bf16 进行训练，建议加载 bf16 格式的权重，以减少数据类型转换带来的消耗和精度损失;
