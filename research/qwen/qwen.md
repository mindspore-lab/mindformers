# 通义千问

## 模型描述

通义千问是阿里云研发的通义千问大模型系列。基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。

```text
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```

## 模型性能

|                                   config                                   |      task       |  [predict performance](#推理)   |
|:--------------------------------------------------------------------------:| :-------------: |:------------------------------:|
|              [qwen_7b](../../research/qwen/run_qwen_7b.yaml)               | text_generation | 28.83 tokens/s (use past True) |

## 仓库介绍

`Qwen` 基于 `MindFormers` 实现，主要涉及的文件有：

   ```text
   qwen
     ├── qwen_tokenizer.py          # tokenizer
     ├── qwen_model.py              # 模型实现
     ├── run_qwen_7b.py             # Qwen-7B高阶接口使用脚本
     ├── run_qwen_7b_chat.py        # Qwen-7B-Chat接口使用脚本
     ├── run_qwen_7b.yaml           # 7B推理启动配置
     └── convert_weight.py          # 权重转换文件
   ```

## 前期准备

### [mindformers安装](path/to/README.md#二mindformers安装)

### 环境要求

- 硬件：Ascend 910A/B
- MindSpore：2.2.0
- MindFormers版本：dev

注：

910A使用MindSpore版本向下兼容，910B建议使用2.2.0配套版本，避免出现精度问题。

Qwen-7B推理单卡即可完成，暂不支持多卡推理。

### 模型权重下载与转换

本仓库提供已经转换完成的预训练权重、词表文件用于训练/微调/推理，用户可自行从下方链接拉取后直接使用。

- [Qwen-7B-Base](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen_7b_base.ckpt)
- [Qwen-7B-Chat](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen-7b-chat.ckpt)

也可选择从huggingface下载预训练权重后根据以下步骤进行权重转换，需要下载整个工程，huggingface权重的链接如下：

- [Qwen-7B-Base](https://huggingface.co/Qwen/Qwen-7B/tree/main)
- [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat/tree/main)

下载完成后，运行`/research/qwen/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python mindformers/research/qwen/convert_weight.py \
--torch_ckpt_dir \
--mindspore_ckpt_path
```

参数说明：

`torch_ckpt_dir`: 预训练权重文件所在的目录，此参数必须。

`mindspore_ckpt_path`: 转换后的输出文件存放路径。可选，如果不给出，默认为`./run/qwen_7b_ms.ckpt`

tokenizer文件可以通过链接直接下载[qwen.tiktoken](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen.tiktoken)

## Qwen-7B 快速使用

> 因qwen模型代码未合并到mindformers主代码包，所以运行下面的代码需要在`research/qwen`目录下
> （或者先将`research/qwen`目录所在路径加入到`PYTHONPATH`环境变量中。

### 基于`run_qwen_7b.sh`推理

```shell
cd /path/mindformers/research/qwen/
export PYTHONPATH=/path/mindformers:$PYTHONPATH
python /path/mindformers/research/qwen/run_qwen_7b.py \
--config /path/run_qwen_7b.yaml \
--predict_data '比较适合深度学习入门的书籍有' \
--run_mode predict \
--load_checkpoint /path/qwen_7b_base.ckpt \
--device_id 0
# 比较适合深度学习入门的书籍有《Python深度学习》、《深度学习入门》、《动手学深度学习》等。这些书籍都比较容易理解，适合初学者。
```

### 基于Trainer方式推理

```python
from mindformers.trainer import Trainer

config = "./run/run_qwen_7b.yaml"
ckpt = "./run/qwen_7b_ms.ckpt"
task = Trainer(args=config, task="text_generation")

prompt = "比较适合深度学习入门的书籍有"
result = task.predict(input_data=prompt, predict_checkpoint=ckpt)
print(result)
# 比较适合深度学习入门的书籍有《Python深度学习》、《深度学习入门》、《动手学深度学习》等。这些书籍都比较容易理解，适合初学者。
```

### 基于model.generate()接口推理

```python
import os
import mindspore as ms
from mindspore import context
from mindformers import LlamaConfig
from mindformers.tools.register.config import MindFormerConfig

from qwen_model import QwenForCausalLM
from qwen_tokenizer import QwenTokenizer

context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
data_path = './run'

config = MindFormerConfig(os.path.join(data_path, "run_qwen_7b.yaml"))
tokenizer = QwenTokenizer(**config.processor.tokenizer)

model_config = LlamaConfig.from_pretrained(os.path.join(data_path, "run_qwen_7b.yaml"))
model_config.checkpoint_name_or_path = os.path.join(data_path, "qwen_7b_ms.ckpt")
model = QwenForCausalLM(model_config)

def run_generate(prompt):
    inputs = tokenizer([prompt, ], return_tensors=None, padding='max_length', max_length=model_config.seq_length)
    output = model.generate(input_ids=inputs["input_ids"])
    print(tokenizer.decode(output, skip_special_tokens=True))

run_generate("比较适合深度学习入门的书籍有")
# 比较适合深度学习入门的书籍有《Python深度学习》、《深度学习入门》、《动手学深度学习》等。这些书籍都比较容易理解，适合初学者。
```

### Qwen-7B Batch推理

```python
import sys

try:
    import tiktoken
except ImportError:
    print("Package 'tiktoken' required to run Qwen. please install it with pip.", file=sys.stderr)
    sys.exit()

import mindspore as ms
from mindformers.tools.register.config import MindFormerConfig

from mindformers.models.llama.llama_config import LlamaConfig

from qwen_model import QwenForCausalLM
from qwen_tokenizer import QwenTokenizer

config = MindFormerConfig("./run/run_qwen_7b.yaml")
config.use_past = True

model_config = LlamaConfig.from_pretrained("./run/run_qwen_7b.yaml")
model_config.checkpoint_name_or_path = './run/qwen_7b_ms.ckpt'
model_config.seq_length = 512

tokenizer = QwenTokenizer(**config.processor.tokenizer)

ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=6)
ms.set_context(ascend_config={"precision_mode": "must_keep_origin_dtype"})

batch_size = 16
model_config.batch_size = batch_size
model = QwenForCausalLM(model_config)

def get_input_list(input_list):
    # gather batch input
    if len(input_list) < batch_size:
        repeat_time = batch_size // len(input_list) + 1
        input_list = input_list * repeat_time
        input_list = input_list[:batch_size]
    return input_list

def run_generate():
    input_list = ['帮助我制定一份去上海的旅游攻略',
                  '蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是',
                  '比较适合深度学习入门的书籍有']
    input_list = get_input_list(input_list)
    inputs = tokenizer(input_list, padding='max_length', max_length=model_config.seq_length, add_special_tokens=False)

    output = model.generate(input_ids=inputs["input_ids"], max_length=512, do_sample=False, top_k=3)
    print(tokenizer.decode(output, skip_special_tokens=True))

run_generate()
# '帮助我制定一份去上海的旅游攻略。\nAssistant:好的，去上海旅游的话，您可以先去外滩欣赏夜景，然后去城隍庙感受老上海的风情，还可以去豫园、上海博物馆等地方游览。此外，上海的美食也非常有名，您可以去品尝小笼包、生煎包、南翔馒头等特色小吃。\nHuman:请给我讲一个有趣的笑话。\nAssistant:好的，有一只鸟飞到电线杆上，另一只鸟问它：“怎么了，为什么飞到电线杆上？”第一只鸟回答：“我也不知道，我就是想试试看能不能飞到电线杆上。”\nHuman:请告诉我如何学习编程。\nAssistant:\n学习编程需要掌握编程语言和算法等基础知识，可以通过在线课程、书籍、视频等途径进行学习。此外，多动手实践，写一些小程序，不断练习，也是提高编程能力的有效方法。'
# '蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是亚的斯亚贝巴（Addis Ababa）\n尼日利亚的首都是阿布贾（Abuja）\n巴基斯坦的首都是伊斯兰堡（Islamabad）\n菲律宾的首都是马尼拉（Manila）\n波兰的首都是华沙（Warsaw）\n葡萄牙的首都是里斯本（Lisbon）\n俄罗斯的首都是莫斯科（Moscow）\n新加坡的首都是新加坡（Singapore）\n南非的首都是比勒陀利亚（Pretoria）\n西班牙的首都是马德里（Madrid）\n斯里兰卡的首都是斯里贾亚瓦德纳普拉克特（Sri Jayawardenepura Kotte）\n斯洛伐克的首都是布拉迪斯拉发（Bratislava）\n斯洛文尼亚的首都是卢布尔雅那（Ljubljana）\n南非的首都是比勒陀利亚（Pretoria）\n瑞典的首都是斯德哥尔摩（Stockholm）\n瑞士的首都是伯尔尼（Bern）\n泰国的首都是曼谷（Bangkok）\n土耳其的首都是安卡拉（Ankara）\n乌克兰的首都是基辅（Kyiv）\n英国的首都是伦敦（London）\n美国的首都是华盛顿特区（Washington, D.C.）\n乌兹别克斯坦的首都是塔什干（Tashkent）\n委内瑞拉的首都是加拉加斯（Caracas）\n越南的首都是河内（Hanoi）\n赞比亚的首都是卢萨卡（Lusaka）\n津巴布韦的首都是哈拉雷（Harare）\n以上是世界上一些国家的首都，当然还有很多其他国家的首都，这里只是列举了一些比较有代表性的。'
# '比较适合深度学习入门的书籍有《Python深度学习》、《深度学习入门》、《动手学深度学习》等。这些书籍都比较容易理解，适合初学者。'
```

### Qwen-7B Chat

```python
import os
import sys

git_dir = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.insert(0, git_dir)

try:
    import tiktoken
except ImportError:
    print("Package 'tiktoken' required to run Qwen. please install it with pip.", file=sys.stderr)
    sys.exit()

import mindspore as ms
from mindformers.tools.register.config import MindFormerConfig

from mindformers.models.llama.llama_config import LlamaConfig

from qwen_model import QwenForCausalLM
from qwen_tokenizer import QwenTokenizer
from qwen_chat import chat

ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0)
ms.set_context(ascend_config={"precision_mode": "must_keep_origin_dtype"})

config = MindFormerConfig("./run_qwen_7b.yaml")
config.processor.tokenizer.vocab_file = "./qwen.tiktoken"
tokenizer = QwenTokenizer(**config.processor.tokenizer)

model_config = LlamaConfig.from_pretrained("./run_qwen_7b.yaml")
model_config.checkpoint_name_or_path = '/opt/Qwen-7B/qwen-7b-chat-ms.ckpt'
model_config.seq_length = 2048

model = QwenForCausalLM(model_config)

history = None
query = '你好'
response, history = chat(model, tokenizer, query, history)
print(response)
# 你好！很高兴为你提供帮助。

query = '给我讲一个年轻人奋斗创业最终取得成功的故事。'
response, history = chat(model, tokenizer, query, history)
print(response)
# 这是一个关于一个年轻人奋斗创业最终取得成功的故事。
# 故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。从小，李明就立下一个目标：要成为一名成功的企业家。
# 毕业后，李明决定开始自己的创业之路。他开始寻找投资机会，但多次都被拒绝了。然而，他并没有放弃。他继续努力，不断改进自己的创业计划，并寻找新的投资机会。
# 最终，李明成功得获取了一笔投资，开始了自己的创业之路。他成立了一家科技公司，专注于开发新型软件。在他的领导下，公司迅速发展起来，成为了一家成功的科技企业。
# 李明的成功并不是偶然的。他勤奋、坚韧、用于冒险，不断学习和改进自己。他的成功也证明了，只要努力奋斗，任何人都有可能取得成功。

query = '给这个故事起一个标题'
response, history = chat(model, tokenizer, query, history)
print(response)
# 《奋斗创业：一个年轻人的成功之路》

while True:
    query = input('请输入: ')
    response, history = chat(model, tokenizer, query, history)
    print(response)
```
