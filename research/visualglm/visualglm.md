# VisualGLM

VisualGLM是由清华大学的GLM团队推出的一个新的多模态对话语言模型，支持图像、中文和英文的输入和输出。VisualGLM大幅度地提升了多模态对话的SOTA水平，创造了令人惊叹的效果，能够根据图像和文本的内容生成符合人类偏好的回答，成为了多模态领域的新时代引领者。 VisualGLM完全开源可商用，基于 Transformer 结构，语言模型部分基于 ChatGLM-6B ，具有 62 亿参数；图像部分通过训练 BLIP2-Qformer 构建起视觉模型与语言模型的桥梁，整体模型共 78 亿参数。

## VisualGLM-6B

VisualGLM-6B 依靠来自于 CogView 数据集的30M高质量中文图文对，与300M经过筛选的英文图文对进行预训练，中英文权重相同。该训练方式较好地将视觉信息对齐到ChatGLM的语义空间；之后的微调阶段，模型在长视觉问答数据上训练，以生成符合人类偏好的答案。

## 前期准备

### 安装mindformers

参考[README](../../README.md) "mindformers安装" 安装mindformers。

### 环境要求

- 硬件: Atlas 800T A2
- MindSpore: 2.2.10
- MindSpore Lite: 2.2.10
- MindFormers: dev
- Mindpet: 1.0.2

**注：** VisualGLM-6B推理可以在单卡上完成部署，全量微调至少需要8卡。

### 生成RANK_TABLE_FILE(多卡运行必须环节)

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注：** 若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成

### VisualGLM-6B 预训练权重下载和转换

#### 1. 从huggingface下载tokenizer权重后转换

从HuggingFace网站下载visualglm 6b词库的文件 ice_text.model。
下载地址：https://huggingface.co/THUDM/visualglm-6b/tree/main

#### 2. 从SAT仓库下载visualglm权重

推荐使用rclone工具下载模型

**步骤**

1) 下载rclone工具
下载地址：<https://rclone.org/downloads/>
根据服务器的类型和处理器，选择对应的文件。
下载完以后解压，把其中的脚本拷贝出来，放到执行目录下：
cp rclone*/rclone /usr/local/bin/

2) 创建rclone配置文件

在home目录创建rclone.conf文件

- Windows系统对于的目录：C:\Users\用户名\.config\rclone\rclone.conf;
- linux系统对应的目录：~/.config/rclone/rclone.conf

配置内容，这里的配置不需要修改：

```text
[r2]
type = s3
provider = Cloudflare
access_key_id = eb4d69e273848089c7f9b9599cdcd983
secret_access_key = 367e9b21fef313f187026320016962b47b74ca4ada7d64d551c43c51e195d7a5
endpoint = https://c8a00746a80e06c4632028e37de24d6e.r2.cloudflarestorage.com
acl = private
```

3) 使用rclone脚本来下载权重文件

**参数说明**

- THREAD_COUNT：下载的线程数量，可以根据实际带宽来调整。

```shell
cd 模型下载路径/
rclone copy  -P --multi-thread-streams THREAD_COUNT  --no-check-certificate -vv --size-only  r2:/sat/visualglm-6b.zip ./
```

4) 执行权重转换脚本

```shell
cd research/visualglm
python convert_weight.py --torch_path TORCH_CKPT_DIR --vit_mindspore_path VIT_CKPT_PATH --qformer_mindspore_path QFORMER_CKPT_PATH --glm_mindspore_path GLM_CKPT_PATH
```

**参数说明**

1. TORCH_CKPT_DIR: huggingface VisualGLM-6B权重保存目录路径，路径要指定到文件；
2. VIT_CKPT_PATH: vit模型mindspore权重文件保存路径，路径要指定到文件；
3. QFORMER_CKPT_PATH: qformer模型mindspore权重文件保存路径，路径要指定到文件；
4. GLM_CKPT_PATH: glm模型mindspore权重文件保存路径和名称，路径要指定到文件。

**注意**:

- 请安装torch=2.0.1和transformers=4.33.2版本，cuda版本11.6及以上
- 该脚本会在glm模型的路径下生成glm_6b_for_lite.ckpt文件，该权重是用于lite推理的。

## MindSpore推理

> 接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)
>

**注意**

- 图片路径：推理用的参考图片在代码仓库的examples路径下
- 提示词：每张图片都有一个对应的参考提示词，可以在example_inputs.jsonl文件找到

由于visualglm模型的权重需要用户自行下载，因此在启动前，请先行在配置文件中将权重的路径配置完成。
打开配置文件 research/visualglm/run_visualglm_6b_image_to_text_generation.yaml，修改如下：

- 替换/path/to/visualglm_qformer.ckpt为上面转换的qformer权重的实际路径
- 替换/path/to/visualglm_vit.ckpt为上面转换的vit权重的实际路径
- 替换/path/to/glm_6b.ckpt为上面转换的glm权重的实际路径
- 替换/path/to/ice_text.model为上面下载的ice_text.model的实际路径

```yaml
model:
  model_config:
    type: VisualGLMConfig
    #...
    checkpoint_name_or_path: "/path/to/visualglm_qformer.ckpt"  # visualglm qformer weight

    vision_config:
      #...
      checkpoint_name_or_path: "/path/to/visualglm_vit.ckpt"  # visualglm vit weight

    text_config:
      type: GLMConfig
      #...
      checkpoint_name_or_path: "/path/to/glm_6b.ckpt" # visualglm glm weight

processor:
  type: VisualGLMProcessor
  image_processor:
    type: VisualGLMImageProcessor
    image_size: 224  # input image size
  tokenizer:
    #...
    checkpoint_name_or_path: "/path/to/ice_text.model"

```

如果使用增量推理，需要在配置文件中use_past值设置为True。

- generate接口推理：

visualglm的generate接口使用脚本已集成在run_visualglm.py脚本中，运行此脚本命令：

```shell
cd research/visualglm
python run_visualglm.py --config CONFIG_PATH --image_path=IMAGE_PATH --prompt=PROMPT --device_id DEVICE_ID
#运行结果：
#['<img> </img>问:描述这张图片。\n答: 泰坦尼克号 电影截图']
# 运行结果

```

**参数说明**

1. CONFIG_PATH：yaml配置的路径，默认是run_visualglm_6b_image_to_text_generation.yaml
2. IMAGE_PATH：推理图片的路径，比如examples/titanic.jpg
3. PROMPT：提示词，比如"描述这张图片。"，注意要加引号
4. DEVICE_ID：NPU卡的编号，默认是0

- pipeline接口推理

visualglm的pipeline接口推理已集成在run_visualglm_pipeline.py脚本中，运行此脚本命令示例：

```shell
cd research/visualglm
python run_visualglm_pipeline.py --config CONFIG_PATH --device_id DEVICE_ID --batch_size BATCH_SIZE --use_past True --seq_length SEQ_LENGTH \
 --image_path IMAGE_PATH --prompt PROMPT
# 运行结果
#['<img> </img>问:描述这张图片。\n答: 泰坦尼克号 电影截图']

```

**参数说明**

1. CONFIG_PATH：yaml配置的路径，默认是run_visualglm_6b_image_to_text_generation.yaml
2. IMAGE_PATH：推理图片的路径，比如examples/titanic.jpg
3. PROMPT：提示词，比如"描述这张图片。"，注意要加引号
4. BATCH_SIZE: 图片批次的大小，默认是1
5. SEQ_LENGTH: token的长度，默认是32
4. DEVICE_ID：NPU卡的编号，默认是0

## MindSpore 微调

注意：目前lora微调只支持数据并行，不支持半自动并行和自动并行

- **step1. 下载微调数据集**

数据集路径：
https://github.com/THUDM/VisualGLM-6B/blob/main/fewshot-data.zip

下载完以后传到服务器，解压到research/visualglm下面
记录下fewhot-data/dataset.json文件的路径

- **step2. 修改微调配置参数**

修改/research/visualglm/run_visualglm_lora.yaml文件:

1. 修改所有path_to_vocab为ice_text.model词库文件的路径
2. 修改所有path_to_dataset为上面数据集dataset.json文件的路径
3. 修改path_to_qformer为上面转换的qformer权重文件visualglm_qformer.ckpt的路径
4. 修改path_to_vit为上面转换的vit权重文件visualglm_vit.ckpt的路径
5. 修改path_to_glm为上面转换的glm权重文件glm_6b.ckpt的路径

```yaml
train_dataset: &train_dataset
  tokenizer:
    type: ChatGLMTokenizer
    max_length: 2048
    vocab_file: "/path_to_vocab/ice_text.model"
  data_loader:
    type: VisualGLMDataLoader
    dataset_dir: "/path_to_dataset/dataset.json"
    shuffle: False
    file_format: json
    random_mapping: True # if true enlarge original dataset "scale" times
    scale: 1

model:
  model_config:
    type: VisualGLMConfig
    #...
    checkpoint_name_or_path: "/path_to_qformer/visualglm_qformer.ckpt"

    vision_config:
      type: ViTConfig
      #...
      checkpoint_name_or_path: "/path_to_vit/visualglm_vit.ckpt"

    text_config:
      type: GLMConfig
      #...
      checkpoint_name_or_path: "/path_to_glm/glm_6b.ckpt"

processor:
  type: VisualGLMProcessor
  image_processor:
    type: VisualGLMImageProcessor
    image_size: 224  # input image size
  tokenizer:
    type: ChatGLMTokenizer
    max_length: 2048
    vocab_file: "/path_to_vocab/ice_text.model"

```

- **step 3. 启动微调任务，按照以下步骤启动：**

调整learning rate和warmup超参，修改/research/visualglm/run_visualglm_lora.yaml文件，根据实际业务调整下面的超参：

1. learning_rate： 微调的模型学习率不宜设置过大
2. warmup_steps：预热步数，表示在训练开始时逐渐增加学习率的步数。这样做可以避免模型在初始阶段受到过大的梯度干扰，提高模型的泛化能力。
3. num_iters：迭代次数，表示模型在一个epoch中处理数据的次数。一个epoch表示模型遍历整个数据集一次。
4. total_steps：总步数，表示模型在整个训练过程中处理数据的次数。总步数等于epoch数乘以迭代次数。如果设置为-1，表示不限制总步数，只根据epoch数来决定训练的终止条件4。

```yaml
# lr sechdule
lr_schedule:
  type: AnnealingLR
  learning_rate: 0.00001
  warmup_steps: 100
  num_iters: 5000
  total_steps: -1 # -1 means it will load the total steps of the dataset
layer_scale: False
layer_decay: 0.65

```

- **step4. 使用shell命令启动微调**

调用下面的脚本启动微调：

```shell
cd research/visualglm
python run_visualglm_finetune.py --config CONFIG_PATH --graph_mode GRAPH_MODE --batch_size BATCH_SIZE --device_id DEVICE_ID
```

**参数说明**

1. CONFIG_PATH：微调配置，默认是run_visualglm_lora.yaml
2. GRAPH_MODE：图模式编号，默认是0。0：graph模式，1：pynative模式
3. BATCH_SIZE：批次大小，默认是1
4. DEVICE_ID：NPU卡的编号，默认是0

- **step5. 并行训练**

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件
这会生成一个名字为hccl_8p_01234567_XXXX.json的文件

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python mindformers/tools/hccl_tools.py --device_num "[START_ID, END_ID)"
```

参数说明：

- \[START_ID, END_ID\]:  表示卡的范围，START_ID是第一块卡的编号，END_ID是最后一块卡的编号，比如8卡为[0,8)

修改run_visualglm_lora.yaml中的并行参数

- use_parallel: 改为True
- parallel_mode：目前只支持数据并行，值为0
- data_parallel：改为上面卡的数量，比如8卡改成8

```yaml
use_parallel: True
parallel:
  parallel_mode: 0
parallel_config:
  data_parallel: 8
  model_parallel: 1

```

运行run_singlenode.sh脚本来执行多卡训练

1. 把HCCL_JSON_PATH替换为上面生成的hccl json文件的路径
2. \[START_ID, END_ID\]:  表示卡的范围，START_ID是第一块卡的编号，END_ID是最后一块卡的编号，要跟上面RANK_TABLE_FILE的配置保持一致；
3. CARD_COUNT: 表示使用NPU卡的数量，要跟上面RANK_TABLE_FILE的配置保持一致

```shell
cd research/visualglm
bash ../run_singlenode.sh \
"python run_visualglm_finetune.py --config CONFIG_PATH --graph_mode GRAPH_MODE --batch_size BATCH_SIZE" \
HCCL_JSON_PATH [START_ID, END_ID] CARD_COUNT

```

**参数说明**

1. CONFIG_PATH：微调配置，默认是run_visualglm_lora.yaml
2. GRAPH_MODE：图模式编号，默认是0。0：graph模式，1：pynative模式
3. BATCH_SIZE：批次大小，默认是1
4. HCCL_JSON_PATH: 多机多卡HCCL通信的配置，使用上面生成的RANK_TABLE_FILE的路径
5. \[START_ID, END_ID\]:  表示卡的范围，START_ID是第一块卡的编号，END_ID是最后一块卡的编号
6. CARD_COUNT：表示使用NPU卡的数量

**注意**

1. 这里START_ID，END_ID和CARD_COUNT要跟上面RANK_TABLE_FILE的配置保持一致

- **step6. 使用shell命令启动推理**

**注意**

- 图片路径：微调推理用的参考图片在代码仓库的finetune路径下
- 提示词：每张图片都有一个对应的参考提示词，可以在finetune_inputs.jsonl文件找到

调用预先开发好的脚本run_visualglm_with_lora.py，传入相关的图片和提示词，会得到相关的文本。

```shell
python run_visualglm_with_lora.py --lora_checkpoint CHECKPOINT_PATH  --config CONFIG_PATH --image_path=IMAGE_PATH --prompt=PROMPT  --device_id DEVICE_ID
#运行结果：
#['这张图片是雨天的。']
```

**说明**:

1. CHECKPOINT_PATH：训练完以后生成的checkpiont的绝对路径，checkpoint一般会保存在下面的路径下output/checkpoint_trainable/rank_[id]/
2. CONFIG_PATH： 表示yaml配置的路径，默认使用run_visualglm_lora.yaml
3. IMAGE_PATH：表示图片的路径，比如finetune/ghost.jpg
4. PROMPT：表示提示词，比如"这张图片的背景里有什么内容？"，注意外面要加引号
5. DEVICE_ID: 表示NPU卡的编号，默认是0
