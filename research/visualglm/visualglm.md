# VisualGLM

VisualGLM是由清华大学的GLM团队推出的一个新的多模态对话语言模型，支持图像、中文和英文的输入和输出。VisualGLM大幅度地提升了多模态对话的SOTA水平，创造了令人惊叹的效果，能够根据图像和文本的内容生成符合人类偏好的回答，成为了多模态领域的新时代引领者。 VisualGLM完全开源可商用，基于 Transformer 结构，语言模型部分基于 ChatGLM-6B ，具有 62 亿参数；图像部分通过训练 BLIP2-Qformer 构建起视觉模型与语言模型的桥梁，整体模型共 78 亿参数。

## VisualGLM-6B

VisualGLM-6B 依靠来自于 CogView 数据集的30M高质量中文图文对，与300M经过筛选的英文图文对进行预训练，中英文权重相同。该训练方式较好地将视觉信息对齐到ChatGLM的语义空间；之后的微调阶段，模型在长视觉问答数据上训练，以生成符合人类偏好的答案。

## 前期准备

### 安装mindformers

参考[README](../../README.md) "mindformers安装" 安装mindformers。

### 环境要求

- 硬件: Ascend 910B
- MindSpore: 2.2.10
- MindSpore Lite: 2.2.10
- MindFormers: dev
- Mindpet: 1.0.2

**注** VisualGLM-6B推理可以在单卡上完成部署，全量微调至少需要8卡。

### 生成RANK_TABLE_FILE(多卡运行必须环节)

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注** 若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成

### VisualGLM-6B 预训练权重下载和转换

#### 1. 从huggingface下载tokenizer权重后转换

[VisualGLM-6B](https://huggingface.co/THUDM/visualglm-6b/tree/main)
只需要下载ice_text.model文件

#### 2. 从SAT仓库下载visualglm权重

推荐使用rclone工具下载模型

a. 下载rclone工具
下载地址：<https://rclone.org/downloads/>
根据服务器的类型和处理器，选择对应的文件。
下载完以后解压，把其中的脚本拷贝出来，放到执行目录下：
cp rclone*/rclone /usr/local/bin/

b. 创建rclone配置文件
在home目录创建rclone.conf文件
比如：C:\Users\用户名\.config\rclone\rclone.conf
linux下面是~/.config/rclone/rclone.conf

```text
[r2]
type = s3
provider = Cloudflare
access_key_id = eb4d69e273848089c7f9b9599cdcd983
secret_access_key = 367e9b21fef313f187026320016962b47b74ca4ada7d64d551c43c51e195d7a5
endpoint = https://c8a00746a80e06c4632028e37de24d6e.r2.cloudflarestorage.com
acl = private
```

c. 使用rclone脚本来下载权重文件

```shell
cd 模型下载路径/
rclone copy  -P --multi-thread-streams 12  --no-check-certificate -vv --size-only  r2:/sat/visualglm-6b.zip ./
```

d. 执行权重转换脚本

```shell
cd research/visualglm
python convert_weight.py --torch_path TORCH_CKPT_DIR --vit_mindspore_path VIT_CKPT_PATH --qformer_mindspore_path QFORMER_CKPT_PATH --glm_mindspore_path GLM_CKPT_PATH
```

```text
# 参数说明
1. TORCH_CKPT_DIR: huggingface VisualGLM-6B权重保存目录路径，路径要指定到文件；
2. VIT_CKPT_PATH: vit模型mindspore权重文件保存路径，路径要指定到文件；
3. QFORMER_CKPT_PATH: qformer模型mindspore权重文件保存路径，路径要指定到文件；
4. GLM_CKPT_PATH: glm模型mindspore权重文件保存路径和名称，路径要指定到文件。
```

**注**:

1. 请安装torch=2.0.1和transformers=4.33.2版本，cuda版本11.6及以上
2. 该脚本会在glm模型的路径下生成glm_6b_for_lite.ckpt文件，该权重是用于lite推理的。

## MindSpore推理

> 接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)
>

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

visualglm的generate接口使用脚本已集成在run_visualglm.py脚本中，运行此脚本命令示例：

```shell
cd research/visualglm
python run_visualglm.py --config run_visualglm_6b_image_to_text_generation.yaml --image_path=images/titanic.jpg --prompt="描述这张图片。" --use_parallel False --device_id 0
#运行结果：
#['<img> </img>问:描述这张图片。\n答: 泰坦尼克号 电影截图']
# 运行结果

```

- pipeline接口推理

visualglm的pipeline接口推理已集成在run_visualglm_pipeline.py脚本中，运行此脚本命令示例：

```shell
cd research/visualglm
python run_visualglm_pipeline.py --device_id 3 --batch_size 1 --use_past True --seq_length 128 --image_path images/titanic.jpg --prompt "描述这张图片。"
# 运行结果
#['<img> </img>问:描述这张图片。\n答: 泰坦尼克号 电影截图']

```

## MindSpore Lite推理

### ckpt转换为mindir

```shell
# 如果需要使用增量推理，配置文件中use_past设置为True
cd research
python export_lite_model.py --mode export --use_past True --device_id 6 --seq_length 512
```

设置use_past=True后生成的mindir有两个，分别在output/mindir_full_checkpoint和output/mindir_inc_checkpoint目录中。
如果不设置use_past或者use_past=False，则只生成mindir_full_checkpoint目录，后续无法使用增量推理。

### lite推理

- step1. 新建context.cfg配置文件

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

- step2. 配置glm模型路径

替换/path/to/glm_6b_for_lite.ckpt为实际的glm for lite模型的路径

```yaml
model:
  model_config:
    #...
    text_config:
      type: GLMConfig
      #...
      checkpoint_name_or_path: "/path/to/glm_6b_for_lite.ckpt" # visualglm glm weight
```

- step2. 使用shell命令启动推理

visualglm的lite推理已集成在run_visualglm_infer_lite脚本中，运行此脚本命令示例：

```shell
# 如果需要增量推理，使用inc_model_path指定路径，否则不需要
cd research
python run_visualglm_infer_lite.py --full_model_path output/mindir_full_checkpoint_bs_1/rank_0_graph.mindir --inc_model_path output/mindir_inc_checkpoint_bs_1/rank_0_graph.mindir --seq_length 512 --ge_config context.cfg --device_id 7

# 运行结果：
#['<img> </img>问:描述这张图片。\n答: 泰坦尼克号 电影截图']

```
