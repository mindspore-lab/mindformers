# ChatGLM3

## 模型描述

ChatGLM3 是智谱AI和清华大学 KEG 实验室联合发布的新一代对话预训练模型。ChatGLM3-6B 是 ChatGLM3 系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 引入了如下特性：**更强大的基础模型**，**更完整的功能支持**，**更全面的开源序列**

```text
@article{zeng2022glm,
  title={Glm-130b: An open bilingual pre-trained model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and others},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}
```

## 仓库介绍

`chatGLM3-6B` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：

    ```text
    mindformers/models/glm3
    ├── __init__.py
    └── glm3_tokenizer.py        # tokenizer
    ```

  glm3的模型结构和config同glm2

2. 模型配置：

    ```bash
    configs/glm3
    ├── predict_glm3_6b.yaml                              # 在线推理配置文件
    ├── run_glm3_6b_finetune_2k_800T_A2_64G.yaml          # Atlas 800T A2 最佳性能全量微调启动配置
    ├── run_glm3_6b_finetune_800T_A2_64G.yaml             # Atlas 800T A2 ADGEN 全量微调启动配置
    ├── run_glm3_6b_multiturn_finetune_800T_A2_64G.yaml   # Atlas 800T A2 多轮对话全量微调启动配置
    └── run_glm3_6b.yaml                                  # ChatGLM3配置模板
    ```

## 前期准备

### 环境要求

**MindFormers安装**以及**软硬件配套关系**参考[MindFormers安装](../../README.md#二MindFormers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。

### 模型权重下载与转换

开发者可以下载获取官方权重后，通过下面提供的**权重转换脚本**，将官方权重转换为MindSpore权重；或直接使用MindFormers提供的**已转换权重**

1. 使用官方权重进行转换

   克隆glm3-6b代码仓，下载分布式的模型文件。

   ```shell
   git lfs install
   git clone https://huggingface.co/THUDM/chatglm3-6b
   ```

   执行 python 脚本，合并模型权重。

   ```python
   from transformers import AutoTokenizer, AutoModel
   import torch

   tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
   model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)

   with open("pt_model_arch.txt", "w") as fp:
       print(model, file=fp, flush=True)
   with open("pt_ckpt.txt", "w") as fp:
       for name, param in model.named_parameters():
           fp.write(f"{name} {param.shape} {param.dtype}\n")
   torch.save(model.state_dict(), "glm3_6b.pth")
   ```

   执行转换脚本，得到转换后的输出文件`glm3_6b.ckpt`。

   ```python
   import mindspore as ms
   import torch as pt
   from tqdm import tqdm

   pt_ckpt_path = "glm3_6b.pth"
   pt_param = pt.load(pt_ckpt_path)

   type_map = {"torch.float16": "ms.float16",
               "torch.float32": "ms.float32"}
   ms_param = []
   with open("check_pt_ckpt.txt", "w") as fp:
       for k, v in tqdm(pt_param.items()):
           if "word_embeddings.weight" in k:
               k = k.replace("word_embeddings.weight", "embedding_table")
           fp.write(f"{k} {v.shape} {v.dtype}\n")
           ms_param.append({"name": k, "data": ms.Tensor(v.numpy())})

   ms.save_checkpoint(ms_param, "glm3_6b.ckpt")
   ```

2. 获取MindFormers提供的已转换权重

   可通过from_pretrained接口下载，也可直接从下面的链接获取

   [glm3_6b权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm3/glm3_6b.ckpt)

   [tokenizer文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm3/tokenizer.model)

### [分布式训练/微调权重合并](../feature_cards/Transform_Ckpt.md)

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../feature_cards/Transform_Ckpt.md)

- step 1. 获取模型切分策略文件：

在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

> 注：lora微调时需要确认配置文件`parallel context config`中`only_trainable_params`设为`False`，以获取所有参数完整策略。

- step 2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

```shell
python transform_ckpt.py \
--src_ckpt_strategy {path}/output/strategy/ \
--src_ckpt_dir {path}/output/checkpoint/ \
--dst_ckpt_dir {path}/target_checkpoint/ \
--prefix glm3_6b
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

下面以 [ADGEN](https://aclanthology.org/D19-1321.pdf) (广告生成) 数据集为例介绍代码的使用方法

### 数据集准备

#### 输入输出格式数据集

ADGEN 数据集任务为根据输入（content）生成一段广告词（summary）。

```yaml
{"content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳", "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"}
```

参照 [ChatGLM仓库](https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning) 的指引下载处理好的 ADGEN 数据集，目录结构为

```text
AdvertiseGen
  ├── train.json
  └── dev.json
```

修改配置文件 `configs/glm3/run_glm3_6b_finetune*.yaml` 中的以下项：

```yaml
train_dataset: &train_dataset
    dataset_dir: "/path/to/AdvertiseGen/train.json"
    origin_columns: ["content", "summary"]
  tokenizer:
    vocab_file: "/path/to/tokenizer.model"
  input_columns: ["input_ids", "labels"]
  max_source_length: 64
  max_target_length: 127

eval_dataset: &eval_dataset
  data_loader:
    dataset_dir: "/path/to/AdvertiseGen/dev.json"
    origin_columns: ["content", "summary"]
  tokenizer:
    vocab_file: "/path/to/tokenizer.model"
  max_source_length: 256
  max_target_length: 256
```

**注意**：微调时的模型`seq_length`需要等于微调数据集的`max_source_length + max_target_length + 1`。
yaml文件中默认的`seq_length: 192`以及`max_source_length: 64`和`max_target_length: 127`适用于ADGEN数据集，
其他数据集的`seq_length`设置，可以遍历并将数据集转换为token_id，取token_id最大长度，`seq_length`太大影响训练性能，
太小影响训练精度，需要做出权衡。

#### 多轮对话格式数据集

首先，克隆 [ToolAlpaca 数据集](https://github.com/tangqiaoyu/ToolAlpaca)，并下载处理脚本 [format_tool_alpaca.py](https://github.com/THUDM/ChatGLM3/blob/7cd5bc78bd6232d02764b60b33874bb2d63a0df0/finetune_chatmodel_demo/scripts/format_tool_alpaca.py)，然后执行脚本执行脚本：

```shell
python mindformers/tools/format_tool_alpaca.py --path ToolAlpaca/data/train_data.json
```

脚本会在执行目录下生成 formatted_data/tool_alpaca.jsonl

也可以在这里下载处理好的数据集：

[tool_alpaca.jsonl](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm3/tool_alpaca.jsonl)

微调时选择配置文件：`configs/glm3/run_glm3_6b_multiturn_finetune*.yaml`

### 全参微调

全参微调使用 `configs/glm3/run_glm3_6b_finetune*.yaml` 配置文件，配置文件中定义了微调所需的各配置项

修改数据集/模型权重配置路径：

- 数据集：修改 `configs/glm3/run_glm3_6b_finetune*.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `configs/glm3/run_glm3_6b_finetune*.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../feature_cards/Training_Algorithms.md#flash-attention)

#### 单卡微调

由于glm3_6b模型较大，全量微调不支持单卡运行

#### 多卡微调

- 单机多卡

推荐使用msrun方式启动

```bash
cd {mindformers根目录}
bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config configs/glm3/run_glm3_6b_finetune_2k_800T_A2_64G.yaml \
    --run_mode finetune"
```

## 推理

### 基本介绍

　　MindFormers 定位打造训练->微调->部署的端到端大模型工具套件，为了更好性能地部署已经微调训练好的大模型，我们利用MindSpore打造了全新的训推一体高性能推理引擎，保证训练与推理使用同一套脚本，为用户提供了开箱即用的推理部署方案，为用户提供端到端的大模型解决方案，帮助用户使能大模型业务。

　　MindSpore 大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合大算子降低推理时延，有效提升网络吞吐量。

### 基于generate的推理

下面提供一个模型推理样例脚本 `infer.py`

```python
import mindspore as ms
from mindformers import AutoConfig, AutoModel, ChatGLM3Tokenizer

# 指定图模式，指定使用训练卡id
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

# 本地加载方式，配置为权重下载章节的tokenizer文件
tokenizer = ChatGLM3Tokenizer("/path/to/tokenizer.model")

# 自定义修改配置后实例化，配置为yaml文件路径，示例yaml文件为configs/glm3/predict_glm3_6b.yaml
config = AutoConfig.from_pretrained("/path/to/your.yaml")
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型

role="user"

inputs_list=["你好", "请介绍一下华为"]

for input_item in inputs_list:
    history=[]
    inputs = tokenizer.build_chat_input(input_item, history=history, role=role)
    inputs = inputs['input_ids']
    # 首次调用model.generate()进行推理将包含图编译时间，推理性能显示不准确，多次重复调用以获取准确的推理性能
    outputs = model.generate(inputs, do_sample=False, top_k=1, max_length=config.seq_length)
    response = tokenizer.decode(outputs)
    for i, output in enumerate(outputs):
        output = output[len(inputs[i]):]
        response = tokenizer.decode(output)
        print(response)
    # answer 1:
    # 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。

    # answer 2:
    # 华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。该公司也在智能手机、电脑、平板电脑、云计算等领域开展业务,其产品和服务覆盖全球170多个国家和地区。

    # 华为的主要业务包括电信网络设备、智能手机、电脑和消费电子产品。公司在全球范围内有超过190,000名员工,其中约一半以上从事研发工作。华为以其高品质的产品和服务赢得了全球客户的信任和好评,也曾因其领先技术和创新精神而获得多项国际奖项和认可。

    # 然而,华为也面临着来自一些国家政府的安全问题和政治压力,其中包括美国政府对其产品的禁令和限制。华为一直坚称自己的产品是安全的,并采取了一系列措施来确保其产品的安全性和透明度。

```

如果需要进行推理，请修改配置文件中(推理脚本中/path/to/your.yaml)的checkpoint_name_or_path项，配置为权重下载章节下载的权重文件：

  ```yaml
  model:
    model_config:
      checkpoint_name_or_path: "/path/to/glm3_6b.ckpt"
  ```

### 基于generate的多角色推理

下面提供一个模型推理样例。

```python
from copy import deepcopy

import mindspore as ms
from mindformers import AutoConfig, AutoModel, ChatGLM3Tokenizer


def process_response(output, history):
    content = ""
    history = deepcopy(history)
    for response in output.split("<|assistant|>"):
        metadata, content = response.split("\n", maxsplit=1)
        if not metadata.strip():
            content = content.strip()
            history.append({"role": "assistant", "metadata": metadata, "content": content})
            content = content.replace("[[训练时间]]", "2023年")
        else:
            history.append({"role": "assistant", "metadata": metadata, "content": content})
            if history[0]["role"] == "system" and "tools" in history[0]:
                content = "\n".join(content.split("\n")[1:-1])
                def tool_call(**kwargs):
                    return kwargs
                parameters = eval(content)
                content = {"name": metadata.strip(), "parameters": parameters}
            else:
                content = {"name": metadata.strip(), "content": content}
    return content, history


# 指定图模式，指定使用训练卡id
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

# 本地加载方式，配置为权重下载章节的tokenizer文件
tokenizer = ChatGLM3Tokenizer("/path/to/tokenizer.model")

# 自定义修改配置后实例化，配置为yaml文件路径，示例yaml文件为configs/glm3/predict_glm3_6b.yaml
config = AutoConfig.from_pretrained("/path/to/your.yaml")
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型

kwargs={}
gen_kwargs = {"max_length": config.seq_length,"num_beams": 1, "do_sample": False, "top_p": 1,"top_k": 1,
              "temperature": 1,**kwargs}

role="system"
text = "假设你现在是一个导游，请尽可能贴近这个角色回答问题。"
history = []
inputs = tokenizer.build_chat_input(text, history=history, role=role)
inputs = inputs['input_ids']
history.append({'role':role, 'content':text}) # 第一个输入

outputs = model.generate(inputs, **gen_kwargs)
outputs =outputs[0][len(inputs[0]):-1]
response = tokenizer.decode(outputs)
print(response, flush=True)
# 您好，我是您的人工智能助手，也可以是你的导游。请问有什么问题我可以帮您解答呢？
response, history = process_response(response, history)
print('history:', flush=True)
print(history, flush=True)

role="user"
text="我打算1月份去海南玩，可以介绍一下海南有哪些好玩的，好吃的么？"
inputs = tokenizer.build_chat_input(text, history=history, role=role)
inputs = inputs['input_ids']
history.append({'role':role, 'content':text}) # 第二个输入
outputs = model.generate(inputs, **gen_kwargs) #, eos_token_id=eos_token_id)
outputs =outputs[0][len(inputs[0]):-1]
response = tokenizer.decode(outputs)
print(response, flush=True)
#  当然可以！海南是一个风景优美、气候宜人的热带海洋省份，拥有丰富的旅游资源和美食。以下是一些您可能会感兴趣的景点和美食：

# 1. 景点：
# - 海南岛：这是海南最著名的景点之一，拥有美丽的沙滩和热带雨林。
# - 亚龙湾：这是海南最著名的海滩之一，拥有柔软的沙滩和清澈的海水。
# - 南山寺：这是海南最著名的佛教寺庙之一，拥有精美的建筑和悠久的历史。
# - 博鳌亚洲论坛永久会址：这是中国最著名的国际会议中心，位于海南岛东海岸。

# 2. 美食：
# - 海南鸡饭：这是海南最著名的美食之一，由鸡肉、米饭和椰汁组成。
# - 海鲜：海南岛周围的海域拥有丰富的海鲜资源，包括螃蟹、龙虾、鱼类等。
# - 椰子饭：这是海南最著名的传统美食之一，由椰子肉和糯米组成。
# - 海南粉：这是海南的一种传统小吃，由米粉和各种肉类、海鲜、蔬菜组成。

# 希望这些信息对您有所帮助，如果您还有其他问题，请随时问我。
response, history = process_response(response, history)

role="user"
text="哪里适合冲浪和潜水呢？"
inputs = tokenizer.build_chat_input(text, history=history, role=role)

inputs = inputs['input_ids']
history.append({'role':role, 'content':text}) # 第三个输入

outputs = model.generate(inputs, **gen_kwargs)
outputs =outputs[0][len(inputs[0]):-1]
response = tokenizer.decode(outputs)
print(response, flush=True)
#  在海南岛，冲浪和潜水的好去处有很多。以下是一些建议：

# 1. 冲浪：
# - 莺歌海：位于海南岛西南部，是冲浪爱好者的天堂。这里的海浪适中，沙滩漂亮，非常适合冲浪。
# - 三亚：位于海南岛南端，拥有许多优质的冲浪场地，如蜈支洲岛、大小洞天等。

# 2. 潜水：
# - 蜈支洲岛：位于海南岛东南部，是潜水爱好者的天堂。这里的潜水条件优越，拥有丰富的珊瑚礁和海洋生物。
# - 莺歌海：位于海南岛西南部，这里的海水清澈，潜水 visibility 很高，非常适合潜水。

# 当然，还有其他一些景点也适合冲浪和潜水，如海南岛的东海岸、西海岸等。具体选择取决于您的兴趣和经验。希望这些建议对您有所帮助！
response, history = process_response(response, history)

role="user"
text="可以帮我做一份旅游攻略吗？"
inputs = tokenizer.build_chat_input(text, history=history, role=role)
inputs = inputs['input_ids']
history.append({'role':role, 'content':text}) # 第四个输入
outputs = model.generate(inputs, **gen_kwargs)
outputs =outputs[0][len(inputs[0]):-1]
response = tokenizer.decode(outputs)
print(response, flush=True)
#  当然可以！以下是一份简要的海南旅游攻略，供您参考：

# 一、行程安排：
# 建议行程为 4-5 天，具体行程可以根据您的需求和时间进行调整。

# 1. 第一天：抵达三亚，入住酒店，适应一下当地的气候和环境。
# 2. 第二天：游览亚龙湾，享受阳光和沙滩，晚上品尝当地的美食。
# 3. 第三天：游览南山寺，感受佛教文化的魅力，晚上可以在三亚市区品尝当地的美食。
# 4. 第四天：前往蜈支洲岛，享受潜水和冲浪的乐趣，晚上返回三亚。
# 5. 第五天：前往博鳌亚洲论坛永久会址，游览博鳌 town，晚上返回三亚，结束行程。

# 二、景点推荐：
# 1. 海南岛：这是海南最著名的景点之一，拥有美丽的沙滩和热带雨林。
# 2. 亚龙湾：这是海南最著名的海滩之一，拥有柔软的沙滩和清澈的海水。
# 3. 南山寺：这是海南最著名的佛教寺庙之一，拥有精美的建筑和悠久的历史。
# 4. 博鳌亚洲论坛永久会址：这是中国最著名的国际会议中心，位于海南岛东海岸。

# 三、美食推荐：
# 1. 海南鸡饭：这是海南最著名的美食之一，由鸡肉、米饭和椰汁组成。
# 2. 海鲜：海南岛周围的海域拥有丰富的海鲜资源，包括螃蟹、龙虾、鱼类等。
# 3. 椰子饭：这是海南最著名的传统美食之一，由椰子肉和糯米组成。
# 4. 海南粉：这是海南的一种传统小吃，由米粉和各种肉类、海鲜、蔬菜组成。

# 四、住宿推荐：
# 建议选择三亚市区的酒店，这样可以方便您游览市区和品尝当地的美食。

# 希望这份攻略对您有所帮助，如果您还有其他问题，请随时问我。
response, history = process_response(response, history)

```

如果需要进行推理，请修改配置文件中(推理脚本中/path/to/your.yaml)的checkpoint_name_or_path项，配置为权重下载章节下载的权重文件：

  ```yaml
  model:
    model_config:
      checkpoint_name_or_path: "/path/to/glm3_6b.ckpt"
  ```

## Q & A

### Q1: 网络训练 loss 不下降、网络训练溢出、`overflow_cond=True` 怎么办？

A1: 执行训练前设置环境变量：

```shell
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
```

重新启动训练。

### Q2: 推理速度非常慢、Mindspore只能跑在CPU上、报错中含有 `te`、`tbe`、`tvm`等字样？

A2: 一般是 Mindspore + Ascend 环境安装问题，确认环境安装过程参照
[安装指南](https://www.mindspore.cn/install/#%E6%89%8B%E5%8A%A8%E5%AE%89%E8%A3%85)并且成功设置了环境变量。执行：

```shell
python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
```

假如执行输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
```

并且没有报错，则说明成功安装了环境。

或许你想问，有没有更方便的环境安装方式？恭喜你，有的，我们还提供现成的
[docker镜像](http://mirrors.cn-central-221.ovaijisuan.com/mirrors.html)，可以依据需求自行取用。

### Q3: Sync stream Failed、exec graph xxx failed？

A3:这类报错较为宽泛，可以打开昇腾host日志进一步定位。

```shell
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

打开昇腾host日志后模型性能将明显下降，定位问题结束后需要取消昇腾日志：

```shell
unset ASCEND_GLOBAL_EVENT_ENABLE ASCEND_GLOBAL_LOG_LEVEL ASCEND_SLOG_PRINT_TO_STDOUT
```

### Q4: the strategy is xxxxxx, shape xxxx cannot be divisible by value x

A4: 检查模型句长是否满足 `max_source_length + max_target_length + 1 = seq_length` 的要求。

### 仍然有疑问？欢迎向我们提出issue，我们将尽快为您解决

提问时麻烦提供以下信息：

1. 执行命令
2. 运行环境，包括硬件版本、CANN版本、Mindspore版本、Mindformers版本
3. 报错完整日志
