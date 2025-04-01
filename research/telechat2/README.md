# 星辰语义大模型 TeleChat2

## 模型描述

- 星辰语义大模型**TeleChat2**是由中国电信人工智能研究院研发训练的大语言模型，包含7B, 35B, 115B三种规模，该系列模型**完全基于国产算力**训练。
- 本次开源**TeleChat2-115B**模型采用10万亿 Tokens中英文高质量语料进行训练，同步开源对话模型**TeleChat2-115B**的多格式、多平台权重文件。
- **TeleChat2**在训练数据、训练方法等方面进行了改进，在通用问答和知识类、代码类、数学类榜单上相比**TeleChat1**均有大幅提升。
    - **TeleChat2**完全基于国产算力和国产深度学习框架进行训练，算力和算法框架更自主可控。优化MP、PP、SP实现方式提升模型性能，优化算子来提升训练速度。
    - 我们使用大量小模型实验来验证scaling law规律，在不同模型结构、不同数据配比和数据清洗方式中寻找最优设计。
    - 采用RingAttention及其他序列切分方式，实现长文训练性能提升；通过ntk-aware+attention-scaling的方式保证训练长度切换时的平稳过渡，以此来保证模型在不同长度数据下的训练效果。
- 在微调数据方面，我们进行了指令复杂性提升与多样性扩充，通过数据合成和人工标注生成高质量数据，并使用拒绝采样生成多样的推理路径；通过研究一套基于base模型反向选择偏好对齐数据方案，基于适配数据最大限度提升模型效果。
    - 通用能力较TeleChat1系列模型提升超过29%，在逻辑推理、总结摘要、长文写作和数学计算上均有大幅提升。

基于GPU，Torch版本的TeleChat2链接：

[TeleChat2](https://github.com/Tele-AI/TeleChat2)

[TeleChat Technical Report](https://arxiv.org/abs/2401.03804)

``` text
@article{wang2024telechat,
      title={TeleChat Technical Report},
      author={Zihan Wang and Xinzhang Liu and Shixuan Liu and Yitong Yao and Yuyao Huang and Zhongjiang He and Xuelong Li and Yongxiang Li and Zhonghao Che and Zhaoxi Zhang and Yan Wang and Xin Wang and Luwen Pu and Huihan Xu and Ruiyu Fang and Yu Zhao and Jie Zhang and Xiaomeng Huang and Zhilong Lu and Jiaxin Peng and Wenjun Zheng and Shiquan Wang and Bingkai Yang and Xuewei he and Zhuoru Jiang and Qiyi Xie and Yanhan Zhang and Zhongqiu Li and Lingling Shi and Weiwei Fu and Yin Zhang and Zilu Huang and Sishi Xiong and Yuxiang Zhang and Chao Wang and Shuangyong Song},
      journal={arXiv preprint arXiv:2401.03804},
      year={2024}
}
```

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

TeleChat2-7b:

| config                                              | task                  | Datasets   | SeqLength | phase           | performance  |
|:---------------------------------------------------:| :-------------------: |:----------:|:---------:|:---------------:|:------------:|
| [TeleChat2_7b](./telechat2-7b/finetune_telechat_7b.yaml) | text_generation       | example_dataset | 8192      | [finetune](#微调) | 2950 tokens/s/p |
| [TeleChat2_7b](./telechat2-7b/predict_telechat_7b.yaml) | text_generation       | example_dataset     | 8192      | [predict](#推理)  | 54.1 tokens/s   |

TeleChat2-35b:

| config                                              | task                  | Datasets   | SeqLength | phase           | performance  |
|-----------------------------------------------------| --------------------- |------------|-----------|-----------------|--------------|
| [TeleChat2_35b](./telechat2-35b/finetune_telechat_35b.yaml) | text_generation       | example_dataset | 8192      | [finetune](#微调) | 516 tokens/s/p |
| [TeleChat2_35b](./telechat2-35b/predict_telechat_35b.yaml) | text_generation       | example_dataset     | 8192      | [predict](#推理)  | 27.7 tokens/s   |

TeleChat2-115b:

| config                                              | task                  | Datasets   | SeqLength | phase           | performance  |
|-----------------------------------------------------| --------------------- |------------|-----------|-----------------|--------------|
| [TeleChat2_115b](./telechat2-115b/finetune_telechat_115b.yaml) | text_generation       | example_dataset | 8192      | [finetune](#微调) | 158 tokens/s/p |
| [TeleChat2_115b](./telechat2-115b/predict_telechat_115b.yaml) | text_generation       | example_dataset     | 8192      | [predict](#推理)  | 26.5 tokens/s   |

TeleChat2-39b-a12b:

| config                                                       | task            | Datasets        | SeqLength | phase            | performance   |
| ------------------------------------------------------------ | --------------- | --------------- | --------- | ---------------- | ------------- |
| [TeleChat2_39b_a12b](./telechat2-39b-a12b/finetune_telechat_39b_a12b.yaml) | text_generation       | example_dataset | 8192      | [finetune](#微调) | 158 tokens/s/p |
| [TeleChat2_39b_a12b](./telechat2-39b-a12b/predict_telechat_39b_a12b_parallel.yaml) | text_generation | example_dataset | 8192      | [predict](#推理) | 36.4 tokens/s |

## 模型文件

`TeleChat2` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/research/telechat2`

   ```text
   telechat
       ├── convert_weight.py                     # torch->ms权重转换脚本
       ├── convert_reversed.py                   # ms->torch权重转换脚本
       ├── telechat_preprocess.py                # telechat模型的mindrecord数据处理脚本
       ├── telechat.py                           # 模型实现
       ├── telechat_config.py                    # 模型配置项
       ├── telechat_layer.py                     # telechat网络层定义
       ├── telechat_interleave.py                # telechat细粒度多副本
       ├── telechat_predict_utils.py             # telechat推理模块
       ├── telechat_tokenizer.py                 # telechat tokenizer
       ├── telechat_transformer.py               # transformer层实现
       └── infer
           ├── telechat.py                       # 推理模型实现（前端并行）
           └── telechat_transformers.py          # 推理模型transformer层实现（前端并行）
   ```

2. 模型配置：`mindformers/research/telechat2`

   ```text
   telechat
       ├── telechat2-7b                                  # telechat2-7B 配置文件
       │   ├── finetune_telechat_7b.yaml                 # 7B全量微调启动配置
       │   ├── predict_telechat_7b.yaml                  # 7B推理启动配置
       │   └── predict_telechat_7b_parallel.yaml         # 7B推理启动配置（前端并行）
       ├── telechat2-35b                                 # telechat2-35B 配置文件
       │   ├── finetune_telechat_35b.yaml                # 35B全量微调启动配置
       │   ├── predict_telechat_35b.yaml                 # 35B推理启动配置
       │   └── predict_telechat_35b_parallel.yaml        # 35B推理启动配置（前端并行）
       ├── telechat2-115b                                # telechat2-115B 配置文件
       │   ├── finetune_telechat_115b.yaml               # 115B全量微调启动配置
       │   ├── predict_telechat_115b.yaml                # 115B推理启动配置
       │   └── predict_telechat_115b_parallel.yaml       # 115B推理启动配置（前端并行）
       └── telechat2-39b-a12b                            # telechat2-39B-A12B 配置文件
           ├── finetune_telechat_39b_a12b.yaml           # 39B-A12B全量微调启动配置
           └── predict_telechat_39b_a12b_parallel.yaml   # 39B-A12B推理启动配置（前端并行）
   ```

3. 任务启动脚本：`mindformers/research/telechat2`

   ```text
   telechat
       ├── run_telechat_predict.py              # 推理脚本
       └── run_telechat.py                      # telechat高阶接口使用脚本
   ```

## 环境及数据准备

### 安装环境

**MindFormers安装**以及**软硬件配套关系**参考[MindFormers安装](../../README_CN.md#二MindFormers安装)和[版本匹配关系](../../README_CN.md#三版本匹配关系)。

### 数据及权重准备

#### 数据集下载

TeleChat2全系列模型中7B，39B-A12, 35B，115B所使用的微调数据集是由中电信人工智能科技有限公司所提供。

step 1. 构建数据集

可构建**单轮对话数据集**或**多轮对话数据集**，数据集为jsonl文件，每条数据内容需包含`input`和`output`字段，参考格式如下：

- 单轮对话数据集

  ```text
  {"input": "<_user>描述一下中国古代文学的典型特点。", "output": "中国古代文学的典型特点包括：\n1. 以散文和诗歌为主要形式：中国古代文学中，散文和诗歌是最为常见和流行的文学形式。\n2. 强调意境：中国古代文学强调意境，而不仅仅关注情节的描述。作家致力于创造一种美感，通过文字表达出美好、深刻和含蓄的意境。\n3. 重视艺术性和审美价值：中国古代文学非常注重艺术性和审美价值，作家用精妙的文字表现出诗意和美感，增强作品的艺术性。\n4. 探究人性和社会问题：中国古代文学经常探究人性和社会问题，通过作品展示出人类在道德和伦理方面的深刻思考，及对社会现象和问题的探讨。\n5. 借物言志：中国古代文学中常借物言志，比如通过自然景观、动植物、传说和神话等形象，表达出人类的情感和内心世界。\n6. 注重传统文化的保护：中国古代文学注重传统文化的保护，作家在其作品中运用了许多古代文学的传统元素，生动展现出中国文化的独特魅力。"}
  ```

- 多轮对话数据集

  ```text
  {"input": "<_user>你能告诉我什么是人工智能吗？<_bot>人工智能是指利用计算机技术和算法，让计算机模拟人类的思维和行为，实现智能化的程序设计。它主要包括机器学习、自然语言处理、计算机视觉等技术和应用。人工智能可以应用于语音识别、图像识别、自动驾驶、智能客服等领域，可以有效地提高工作效率，优化资源配置，带来更好的人机交互体验。<_user>未来的人工智能将会发展成怎样呢？", "output": "未来的人工智能发展方向众多，以下是一些可能的方向：\n\n1. 个性化AI：人们对于个性化服务的需求越来越高，未来的人工智能将更加注重个性化，为用户提供量身定制的体验。\n\n2. 更加智能化的自动化：未来的人工智能将致力于让机器更加智能地自动化完成各种任务，从而提升效率、降低生产成本。\n\n3. 完全智能化的知识处理：未来的人工智能将能够更深入地处理和理解知识，掌握更多的上下文信息，从而能够更智能地应用到真实的场景中。\n\n4. 更强大的计算能力：未来的人工智能将伴随着计算技术的不断发展而不断提高其处理和计算能力，为更加复杂的任务处理提供支持。\n\n5. 更好的人机交互：未来的人工智能将更好地与人类交互，可预见的未来将有更多的机器人出现在人们的生活中，应用到教育、医疗、娱乐等领域。\n\n总之，未来的人工智能将在不断的发展中迎来更多更广泛的应用场景和解决更多的实际问题，使人们的生活更加便捷、高效和智能化。"}
  ```

step 2. 处理数据成mindrecord格式

```bash
# 使用mindformers/research/telechat/telechat_preprocess.py进行数据预处理和Mindrecord数据生成
python telechat_preprocess.py \
--input_dataset_file /{path}/ \
--vocab_file_path /{path}/tokenizer.model \
--max_length 8192 \
--output_path /{path}/
```

```text
# 参数说明
input_dataset_file: 预训练的数据集
vocab_file_path: 词模型文件路径(如使用上述链接下载，指定到对应路径下即可)
max_length: 数据集长度
output_path: 生成数据集的路径
```

  > 注：`bos`, `eos`, `pad`等特殊`ids`要和`yaml`配置文件中`model_config`部分保持一致，默认`bos_token_id=1`, `eos_token_id=2`, `pad_token_id=3`。
如果有所修改，配置文件中对应设置也需要修改，通常预训练数据不包含`pad_token`，因此建议设置`pad_token_id=-1`。

#### 模型权重下载与转换

MindFormers提供已经转换完成的预训练权重、词表文件用于预训练、微调和推理，开发者可以下载获取官方权重后，通过下面提供的**权重转换脚本**，将官方权重转换为MindSpore权重；或直接使用MindFormers提供的**已转换权重**

1.torch模型权重及词模型下载链接：

- [TeleChat2-7b](https://modelscope.cn/models/TeleAI/TeleChat2-7B)
- [TeleChat2-39B-A12B](https://modelscope.cn/models/TeleAI/TeleChat2-39B-A12B)
- [TeleChat2-35b](https://modelscope.cn/models/TeleAI/TeleChat2-35B)
- [TeleChat2-115b](https://modelscope.cn/models/TeleAI/TeleChat2-115B)

下载完成后，运行如下转换脚本，将全量微调的权重转换为完整的ckpt权重。

```shell
python mindformers/research/telechat2/convert_weight.py \
--torch_path TORCH_CKPT_DIR \
--mindspore_path {path} \
```

```text
# 参数说明
torch_path: torch版本权重保存目录路径
mindspore_path: 权重保存文件名，可以指定自定义保存路径
```

2.获取MindFormers提供的已转换权重，可直接从下面的链接获取。

- [TeleChat2-7b](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/Telechat_7B/Telechat_7B.zip)
- [TeleChat2-35b](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/Telechat_35B/Telechat_35B.zip)
- [TeleChat2-115b](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/Telechat_115B/Telechat_115B.zip)
- [Telechat2-39b-a12b](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/Telechat_39B_A12.tar)

### [分布式训练/微调权重合并](../../docs/feature_cards/Transform_Ckpt.md)

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

- step 1. 获取模型切分策略文件：

在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

- step 2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

```shell
python transform_ckpt.py \
--src_ckpt_strategy {path}/output/strategy/ \
--src_ckpt_dir {path}/output/checkpoint/ \
--dst_ckpt_dir {path}/target_checkpoint/ \
--prefix telechat_{size}
```

```text
# 参数说明
src_ckpt_strategy: 步骤1中的切分策略文件路径
src_ckpt_dir: 原切分权重文件夹
dst_ckpt_dir: 目标路径
prefix: ckpt文件前缀名
```

> 注：`transform_checkpoints` 接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以新建conda环境安装mindspore 2.0的cpu版本以执行该脚本

## 微调

MindFormers提供`TeleChat2-115B`的微调示例，过程中使用中电信人工智能科技有限公司提供的数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 全参微调

#### 多机训练

- step 1. 修改模型对应的配置文件。

  在模型对应的配置文件`research/telechat2/finetune_telechat_115b.yaml`中，用户可自行修改模型、训练相关参数(推荐开启flash_attention，可加速训练)，并通过`train_dataset`的`dataset_dir`参数，指定训练数据集的路径。

  1. 增加脚本入参`--load_checkpoint /{path}/telechat_115b.ckpt`加载预训练权重
  2. 设置启动脚本中的`--train_dataset_dir /{path}/dataset.mindrecord`加载微调数据集
  3. 设置启动脚本中的`--run_mode finetune`

  配置文件中各参数含义详见[Config配置说明文档](https://gitee.com/mindspore/mindformers/blob/master/configs/README.md)。auto_parallel说明详见[自动并行](../../docs/feature_cards/Auto_Parallel.md)。

- step 2. 根据服务器节点数等信息，修改相应的配置。

  ```yaml
  # 以telechat-115b模型8机64卡训练为例，默认配置机4096卡，如果节点数有变，需要修改相应的配置。
  # 配置文件路径：finetune_telechat_115b.yaml
  parallel_config:
    data_parallel: 1
    model_parallel: 8
    pipeline_stage: 8
    micro_batch_num: 8
    vocab_emb_dp: True
    gradient_aggregation_group: 4
  ```

- step3. 设置环境变量，变量配置如下：

  ```bash
  export ENABLE_CELL_REUSE=1  #编译加速
  export MS_DEV_SIDE_EFFECT_LOAD_ELIM=3  # 去除TensorMove
  export MS_MEMORY_POOL_RECYCLE=1  # 内存优化
  export GE_NOT_CUT=1   # 内存优化
  ```

- step 4. 执行运行脚本。

  在多机上同时拉起任务，每台机器拉起方式参考单机多卡启动方式。

  ```shell
  cd mindformers/

  # 节点0，节点ip为192.168.1.1，作为主节点，总共16卡且每个节点8卡
  bash scripts/msrun_launcher.sh "python run_mindformer.py \
   --config research/telechat2/finetune_telechat_115b.yaml
   --train_dataset /{path}/dataset.mindrecord \
   --use_parallel True \
   --register_path ./research/telechat2" \
    16 8 192.168.1.1 8118 0 output/msrun_log False 300

  # 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
  bash scripts/msrun_launcher.sh "python run_mindformer.py \
   --config research/telechat2/finetune_telechat_115b.yaml
   --train_dataset /{path}/dataset.mindrecord \
   --use_parallel True \
   --register_path ./research/telechat2" \
    16 8 192.168.1.1 8118 1 output/msrun_log False 300
  ```

  ```text
  # 参数说明
  config: 配置文件路径
  train_dataset: 训练数据集文件夹路径
  use_parallel：开启并行训练
  register_path: 外部模型注册路径
  ```

## 推理

推理时所需的模型词表可在[模型权重下载与转换](#模型权重下载与转换)章节中下载得到，对应文件为`tokenizer.model`。

### 快速推理

运行`run_mindformer.py`启动快速推理。

#### 参数配置

在`predict_telechat_xxx.yaml`中填写`vocab_file`字段

```yaml
processor:
  tokenizer:
    vocab_file: 'path/to/tokenizer.model'
```

#### 启动推理

- 7b模型单卡推理

  ```bash
  cd mindformers/
  python run_mindformer.py \
  --config ./research/telechat2/predict_telechat_7b.yaml \
  --load_checkpoint path/to/ckpt_path \
  --use_parallel False \
  --predict_data "<_start><_user>生抽与老抽的区别？<_bot>" \
  --register_path ./research/telechat2
  ```

- 39b-a12模型2卡推理

```bash
cd mindformers/
bash scripts/msrun_launcher.sh "python run_mindformer.py \
--config ./research/telechat2/predict_telechat_39b_a12b.yaml \
--load_checkpoint path/to/ckpt_path \
--predict_data '<_start><_user>生抽与老抽的区别？<_bot>' \
--auto_trans_ckpt True \
--use_parallel True \
--register_path ./research/telechat2 2
```

- 35b模型2卡推理

  默认使用完整权重，开启权重自动转换`auto_trans_ckpt=True`。

  ```bash
  cd mindformers/
  bash scripts/msrun_launcher.sh "python run_mindformer.py \
  --config ./research/telechat2/predict_telechat_35b.yaml \
  --load_checkpoint path/to/ckpt_path \
  --predict_data '<_start><_user>生抽与老抽的区别？<_bot>' \
  --auto_trans_ckpt True \
  --use_parallel True \
  --register_path ./research/telechat2" 2
  ```

- 115b模型8卡推理

  默认使用完整权重，开启权重自动转换`auto_trans_ckpt=True`。

  ```bash
  cd mindformers/
  bash scripts/msrun_launcher.sh "python run_mindformer.py \
  --config ./research/telechat2/predict_telechat_115b.yaml \
  --load_checkpoint path/to/ckpt_path \
  --predict_data '<_start><_user>生抽与老抽的区别？<_bot>' \
  --auto_trans_ckpt True \
  --use_parallel True \
  --register_path ./research/telechat2" 8
  ```

- 参数说明

  ```text
  config: 模型的配置文件
  load_checkpoint: 权重路径
  predict_data: 输入的问题
  auto_tans_ckpt: 权重自动转换开关
  use_parallel: 并行模式开关
  register_path: 外部模型注册路径
  ```

#### 推理结果

115B 模型推理结果如下：

```text
生抽与老抽的区别？

生抽和老抽是两种不同的酱油，它们在风味、色泽和用途上都有所区别。

1.颜色：生抽的颜色比较淡，而老抽的颜色较深。生抽的颜色呈红褐色或棕红色，而老抽的颜色则呈棕黑色。

2.味道：生抽具有鲜美的咸味和微甜的味浅，而老抽浓郁，颜色较深。根据个人口味和烹饪需求选择不同的酱油类型可以获得更好的口感和菜肴效果。
```

### 在线推理

Telechat2提供了专用推理脚本，运行`research/telechat2/run_telechat_predict.py`启动在线推理。

#### 单卡推理

```bash
cd mindformers/
python research/telechat2/run_telechat_predict.py \
--yaml_file path/to/yaml_file \
--vocab_file_path path/to/tokenizer.model \
--checkpoint_path path/to/ckpt_path \
--use_parallel False
```

#### 多卡推理

默认使用完整权重，开启权重自动转换`auto_trans_ckpt=True`，修改yaml文件中的`model_parallel`参数为实际运行卡数后，启动多卡推理。

```bash
cd mindformers/
bash scripts/msrun_launcher.sh "python research/telechat2/run_telechat_predict.py \
--yaml_file path/to/yaml_file \
--vocab_file_path path/to/tokenizer.model \
--checkpoint_path path/to/ckpt_path \
--use_parallel True \
--auto_trans_ckpt True" 卡数
```

### 长序列推理

Telechat2的前端并行推理代码（`research/telechat2/infer/telechat.py`）已适配DynamicNTK算法，以实现训短推长的效果。

#### 参数配置

Telechat2原生支持8K长度推理，以支持最长16K推理为例，修改`research/telechat2/infer`下的`predict_telechat_xxx.yaml`。

```yaml
seq_length: 16384              # 最大推理长度
max_position_embedding: 8192   # 模型原支持长度
extend_method: "DYNAMIC_NTK"   # 外推模式设置
block_size: 16                 # 每块block的大小，建议固定设置为16
num_blocks: 1024               # block总数，确保num_blocks * block_size ≥ seq_length
```

#### 单卡推理

使用`research/telechat2/infer/run_telechat_predict.py`启动在线推理。

```bash
cd mindformers/
python research/telechat2/infer/run_telechat_predict.py \
--input_txt ./research/telechat2/infer/xiyou.txt \
--yaml_file path/to/yaml_file \
--vocab_file_path path/to/tokenizer.model \
--checkpoint_path path/to/ckpt_path \
--use_parallel False
```

#### 多卡推理

默认使用完整权重，开启权重自动转换`auto_trans_ckpt=True`，修改yaml文件中的`model_parallel`参数为实际运行卡数后，启动多卡推理。

```bash
cd mindformers/
bash scripts/msrun_launcher.sh "python research/telechat2/infer/run_telechat_predict.py \
--input_txt ./research/telechat2/infer/xiyou.txt \
--yaml_file path/to/yaml_file \
--vocab_file_path path/to/tokenizer.model \
--checkpoint_path path/to/ckpt_path \
--use_parallel True \
--auto_trans_ckpt True" 卡数
```

### MindIE部署

MindIE是基于昇腾硬件的运行加速、调试调优、快速迁移部署的高性能深度学习推理框架。

Telechat2主要采用**前端并行推理脚本**来部署mindie服务化推理，以Telechat2-7b为例，步骤参考如下：

- **构建推理文件夹**

  ```bash
  # 创建模型文件夹
  mkdir model_path
  # 拷贝yaml配置文件和模型相关文件到模型文件夹
  cp research/telechat2/infer/predict_telechat2_7b.yaml model_path
  cp research/telechat2/infer/telechat.py model_path
  cp research/telechat2/telechat_tokenizer.py model_path
  cp research/telechat2/telechat_config.py model_path
  # 生成config.json
  cp research/telechat2/get_config.py model_path
  cd model_path
  python get_config.py
  ```

  文件夹结构如下所示

  ```text
  model_path
   ├── config.json                # 模型json配置文件
   ├── predict_telechat2_7b.yaml  # 模型yaml配置文件
   ├── telechat.py                # 模型文件
   ├── telechat_tokenizer.py      # 模型词表文件
   └── telechat_config.py         # 模型配置文件
  ```

- **权重转为qkv_concat格式（可选）**

  将权重由kv_concat转为qkv_concat格式，可以提升推理性能2%左右，转换命令如下：

  ```bash
  python convert_weight.py --model_name=telechat_7B --qkv_concat=True --pre_ckpt_path=/path/to/telechat2_7b.ckpt  --mindspore_ckpt_path=/path/to/telechat2_7b_qkv.ckpt
  ```

- **配置yaml文件**

  在yaml中配置`load_checkpoint`权重路径和`vocab_file`词表模型路径，如果使用qkv_concat格式权重，设置`qkv_concat`为True。

  ```yaml
  load_checkpoint: /path/to/telechat2_7b_qkv.ckpt
  model:
    model_config:
      qkv_concat: True
  processor:
    tokenizer:
      vocab_file: /path/to/tokenizer.model
  ```

- **配置mindie参数**

  ```bash
  cd /usr/local/Ascend/mindie/latest/mindie-service
  vim conf/config.json
  ```

  配置以下参数：

  ```yaml
  modelWeightPath: 'model_path'   # 模型路径
  npuDeviceIds: [[0]]             # 使用卡号
  worldSize: 1                    # 卡数
  backendType: 'ms'               # 推理后端
  maxSeqLen: 8192                 # 输入+输出最大长度
  maxInputTokenLen: 6144          # 最大输入长度
  npuMemSize: 8                   # KVCache显存占用
  httpsEnabled: False             # 使能https开关
  ```

- **启动mindie**

  ```bash
  ./bin/mindieservice_daemon > mindie.log 2>&1 &
  tail -f mindie.log
  ```

  当log日志中出现**`Daemon start success!`**，表示服务启动成功！

- **推理服务验证**

  发送推理请求

  ```bash
  curl -w "\ntime_total=%{time_total}\n" -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"inputs": "生抽与老抽的区别？", "parameters": {"do_sample": false, "max_new_tokens": 512}, "stream": false}' http://127.0.0.1:1025/generate &
  ```

  返回如下推理结果，表示推理服务正常

  ```text
  {"generated_text":"<_bot>抽抽与老抽都是指在烹饪过程中用于增加食物色泽和风味的调料，它们之间主要有以下几个区别：\n\n1. **成分不同**：\n - 老抽：老抽是由生抽（酱油）经过发酵后形成的，主要成分包括水、糖、盐、谷氨酸、甘氨酸、氨基酸、核苷酸等。\n - 抽抽：抽抽是一种调味剂，通常是由各种香料和调味料调配而成，成分可能包括盐、胡椒粉、辣椒粉、大蒜粉、洋葱粉等。\n\n2. **色泽和口味差异**：\n - 老抽：老抽的颜色一般较深，呈深棕色或黑褐色，口味比较浓郁，带有甜味和咸味，适合用于上色和增加菜肴的醇香。\n - 抽抽：抽抽的颜色和口味较为多样，通常较为清淡，可以用于提味或增加风味，但不具备上色功能。\n\n3. **使用时机不同**：\n - 老抽：由于上色效果较好，通常用于红烧、炖菜等需要颜色的菜肴，或者在炒菜时作为调味使用。\n - 抽抽：抽抽多用于调味，可以提鲜、增加风味，适用于多种菜肴，但通常不直接用于上色。\n\n4. **烹饪效果不同**：\n - 老抽：使用老抽上色可以使菜肴颜色更加浓郁、诱人，增加食欲。\n - 抽抽：抽抽主要用于调味，使菜肴味道更加丰富，但单独使用上色效果不如老抽。\n\n综上所述，老抽和抽抽在成分、色泽、口味、使用时机和烹饪效果等方面都有所不同。在实际烹饪中，可以根据菜肴的需求和个人口味选择合适的调料进行调配。<_end>"}
  time_total=5.099106
  ```







