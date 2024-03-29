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
   qwen
     ├── run_qwenvl_stage1_910b.yaml    # qwenvl第一阶段训练启动配置文件
     ├── run_qwenvl_stage2_910b.yaml    # qwenvl第二阶段训练启动配置文件
     └── run_qwenvl_stage3_910b.yaml    # qwenvl第三阶段微调启动配置文件
   ```

3. 环境准备和任务启动脚本：

   ```text
   qwen
     ├── qwenvl_dataset.py        # 数据集加载 
     ├── qwenvl_dataloader.py     # 数据集加载 
     ├── qwenvl_transform.py      # 数据加载时使用的数据转换
     ├── qwenvl_processor.py      # 推理时候使用的数据处理
     ├── convert_weight.py        # 权重转换脚本
     └── run_qwenvl.py            # Qwen高阶接口脚本
   ```

## 前期准备

### 环境搭建

- 硬件：Atlas 800T A2
- MindSpore：2.2.11
- Python：3.8+

#### 从零搭建

1. 环境搭建参考 [MindSpore官网](https://www.mindspore.cn/install/)，安装MindSpore2.2.11 + CANN社区版7.0.0.beta1配套版本；
2. 使用命令安装mindformers

```shell
cd mindformers
bash build.sh
```

#### 使用镜像

- docker镜像拉取命令

```shell
docker pull swr.cn-central-221.ovaijisuan.com/mindformers/mindformers1.0_mindspore2.2.11:aarch_20240125
```

- 创建容器

```shell
# --device用于控制指定容器的运行NPU卡号和范围
# -v 用于映射容器外的目录
# --name 用于自定义容器名称

docker run -it -u root \
--ipc=host \
--network host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /etc/localtime:/etc/localtime \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /var/log/npu/:/usr/slog \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
--name {请手动输入容器名称} \
swr.cn-central-221.ovaijisuan.com/mindformers/mindformers1.0_mindspore2.2.11:aarch_20240125 \
/bin/bash
```

- 容器内重新安装mindformers

```shell
pip uninstall mindformers

cd mindformers
bash build.sh
```

### RANK_TABLE_FILE准备

运行`mindformers/tools/hccl_tools.py`，生成`RANK_TABLE_FILE`文件

```shell
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
        {
          "device_id": "0",
          "device_ip": "192.1.27.6",
          "rank_id": "0"
        },
        {
          "device_id": "1",
          "device_ip": "192.2.27.6",
          "rank_id": "1"
        },
        {
          "device_id": "2",
          "device_ip": "192.3.27.6",
          "rank_id": "2"
        },
        {
          "device_id": "3",
          "device_ip": "192.4.27.6",
          "rank_id": "3"
        },
        {
          "device_id": "4",
          "device_ip": "192.1.27.7",
          "rank_id": "4"
        },
        {
          "device_id": "5",
          "device_ip": "192.2.27.7",
          "rank_id": "5"
        },
        {
          "device_id": "6",
          "device_ip": "192.3.27.7",
          "rank_id": "6"
        },
        {
          "device_id": "7",
          "device_ip": "192.4.27.7",
          "rank_id": "7"
        }
      ],
      "host_nic_ip": "reserve"
    }
  ],
  "status": "completed"
}
```

### 数据集准备以及数据集相关参数解释
#### 参数解释
```json
data_loader:
    type: QwenVLDataLoader
    dataset_dir: "/path/to/dataset"
    image_dir: ["images"]
    stage: 3
    column_names: ["image", "text"]
    shuffle: True
    task_config:
      sft:
        annotation_files: [ "multi-round-chat/qwenvl_stage3_data.json" ]
    extra_kwargs:
      max_img_len: 3
      map_function_kwargs:
        user_role_name: user
        assistant_role_name: assistant
......
text_transforms:
    type: QwenVLTransform
    max_length: 2049
```
data_loader的参数解释：
- type: 数据集加载器的类型，固定为QwenVLDataLoader。
- dataset_dir: 数据集的路径。
- image_dir: 图片的路径。当数据集包含`<img>xxx.jpg</img>`时，会从`/path/to/dataset/images/xxx.jpg`这个路径去获取图片。
- stage：对应QwenVL的不同stage。
- column_names: 数据集输出的列名。一般为image和text
- shuffle: 是否打乱数据集。
- task_config: 任务配置。其中stage3对应sft名称以及其annotation_files的相对路径。如上的json含义为会从`/path/to/dataset/multi-round-chat/qwenvl_stage3_data.json`中读取annotation_files。
- extra_kwargs: 额外的参数。
    - max_img_len: 图片的最大数量。单个对话中的图片不足最大图片数量则会pad到最大数量。
    - map_function_kwargs: map_function的参数。
        - user_role_name: 提出问题方的名称，对应数据集中的from。
        - assistant_role_name: 回答问题方的名称，对应数据集中另外一个的from。

text_transforms的参数解释：
- type: 数据集加载器的类型，固定为QwenVLTransform。
- max_length: 文本的最大长度。会将输出的caption pad到max_length。

#### 数据集准备
目前QwenVL支持的数据集以及格式：Stage1和Stage2支持COCO Caption，COCO VQA_v2。Stage3支持QwenVL github上提供个数据集格式见如下示例：
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
制作stage3数据集时，请按照如上格式。且stage3数据集中的`<img>xxx.jpeg</img>`必须为相对路径，存放位置和参数解释中提及的路径保持一致。


### 模型权重准备

## 权重转换

从 `Huggingface` 下载 [`Qwen-VL` 权重](https://huggingface.co/Qwen/Qwen-VL)。

使用 `convert_weights.py` 脚本转换权重：

```shell
python convert_weights.py --torch_ckpt_dir /path/to/hf/ckpt \
--mindspore_ckpt_path /path/to/qwenvl_ms.ckpt \
--dtype float32 \
--vit_num_head 16
```

`--torch_ckpt_dir` 传入从 `Huggingface` 下载好的权重路径，`--mindspore_ckpt_path`
是用于存储转换后权重的路径。默认转换为 `float32` 的权重。
`vit_num_head` 用于转换 `ViT` 中的 `Transformer` 中的 `Attention` 层权重，默认为 `16`。

## stage3训练：指令微调
可以参考以下脚本执行QwenVL的微调流程。在执行前请按照上一章节所描述的修改数据集的位置。
```shell
cd research
bash run_singlenode.sh "python qwenvl/run_qwenvl.py --config /path/to/run_qwenvl_stage3_910b.yaml --use_parallel True --run_mode finetune --load_checkpoint /path/to/qwenvl.ckpt --vocab_file /path/to/qwen.tiktoken --seq_length 2048 --image_size 448" /path/to/hccl.json '[0,8]' 8
```


## 推理


### 基于高阶接口推理

### 基于Generate推理

修改 `run_qwenvl_stage3_910b.yaml` 中的 `use_past` 配置以启用增量推理：

```yaml
model:
  model_config:
    type: QwenVLConfig

    # ......
    freeze_llm: False
    use_past: True  # <--- add use_past here

  text_config:
      # ......
      param_init_type: "float16"
      use_past: True # <--- change to True here
```

```shell
python run_qwenvl.py --config /path/to/infer.yaml \
--load_checkpoint /path/to/ckpt \
--vocab_file /path/to/qwen.tiktoken \
--device_id 0 --run_mode predict \
--image_path /path/to/demo.jpeg \
--image_size 448 \
--prompt "Describe the image in English:" 
```

例如：

![`demo.jpeg`](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg)
图片链接：https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg

生成结果：

```text
Picture 1: <img>/path/to/demo.jpeg</img>
Describe the image in English: A woman and a dog sitting on the beach.<|endoftext|>
```

### Batch推理

