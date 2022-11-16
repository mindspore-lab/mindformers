# 欢迎来到MindSpore Transformer

## 介绍

MindSpore Transformer套件的目标是构建一个大模型训练、推理、部署的全流程套件：
提供业内主流的Transformer类预训练模型，
涵盖丰富的并行特性。 期望帮助用户轻松的实现大模型训练。

MindSpore Transformer基于MindSpore内置的并行技术，具备如下特点：

- 一行代码实现从单卡到大规模集群训练的无缝切换。
- 提供灵活易用的个性化并行配置。
- 能够自动进行拓扑感知，高效地融合数据并行和模型并行策略；实现单卡到大规模集群的无缝切换。

如果您对MindSpore Transformer有任何建议，请通过Gitee或MindSpore与我们联系，我们将及时处理。

目前支持的模型列表如下：

- BERT
- GPT
- OPT
- T5

### 安装

目前仅支持源码编译安装，用户可以执行下述的命令进行包的安装

```bash
git clone https://gitee.com/mindspore/transformer.git
cd transformer
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py install
```

#### 版本匹配关系

|版本对应关系| Mindformer  | MindSpore |
|-----------| -----------| ----------|
|版本号      | 0.2.0      | 2.0     |
|版本号      | 0.1.0      | 1.8     |

### 使用

目前该库提供两种方式供用户使用

#### 直接导入模型

用户可以基于`mindtransformer.models`接口，直接导入需要的模型：

```python
from mindtransformer.models import bert
config = bert.BertConfig(num_layers=1, embedding_size=8, num_heads=1)
net = bert.BertModel(config, is_training=False)
```

#### 基于Trainer接口

用户可以基于Trainer方式执行模型的创建和训练：

```python
import numpy as np
from mindspore.dataset import GeneratorDataset
from mindtransformer.trainer import Trainer, TrainingConfig

class GPTTrainer(Trainer):
    """GPT trainer"""
    def build_model(self, model_config):
        from mindtransformer.models.gpt import GPTWithLoss
        my_net = GPTWithLoss(model_config)
        return my_net

    def build_model_config(self):
        from mindtransformer.models.gpt import GPTConfig
        return GPTConfig(num_layers=1, hidden_size=8, num_heads=1, seq_length=14)

    def build_dataset(self):
        def generator():
            data = np.random.randint(low=0, high=15, size=(15,)).astype(np.int32)
            for _ in range(10):
                yield data

        ds = GeneratorDataset(generator, column_names=["text"])
        ds = ds.batch(2)
        return ds

    def build_lr(self):
        return 0.01

trainer = GPTTrainer(TrainingConfig(device_target='CPU', epoch_size=2, sink_size=2))
trainer.train()
```

## 使用指南

目前提供下述的文档

- [使用指南](docs/how_to_config.md)
- [如何使用BERT进行微调](docs/how_to_train_bert.md)

## Benchmark

请[在此](docs/benchmark.md)查看每个模型的复现性能基准。

## FAQ

1. 如何迁移HuggingFace权重 ？

请查看[如何转换HuggingFace的权重](./tools/README.md)

## 贡献

欢迎参与社区贡献，详情参考[Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING_CN.md)。

## 许可证

[Apache 2.0许可证](LICENSE)
