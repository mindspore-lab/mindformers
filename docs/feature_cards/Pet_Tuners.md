# 低参微调

MindPet（Pet：Parameter-Efficient Tuning）是属于Mindspore领域的微调算法套件。随着计算算力不断增加，大模型无限的潜力也被挖掘出来。但随之在应用和训练上带来了巨大的花销，导致商业落地困难。因此，出现一种新的参数高效（parameter-efficient）算法，与标准的全参数微调相比，这些算法仅需要微调小部分参数，可以大大降低计算和存储成本，同时可媲美全参微调的性能。

## [微调支持列表](../model_support_list.md#微调支持列表)

## 使用示例

1. 修改任务模型，以GPT2为例，主要有以下步骤：
    - 继承GPT2LMHeadModel
    - 定义替换lora算法的替换规则`self.pet.pet_config.reg_rules = r'.*dense*|.*linear*|.*mapping*|.*projection*'`
    - 调用lora适配器接口修改预训练模型
    - 导入预训练权重以及冻结预训练模型权重
    - 为了使用MindFormer的训练流程，将GPT2WithLora微调模型注册到MindFormer中

具体代码如下：

```python
@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GPT2WithLora(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config = None, pet = None, **kwargs):
        super().__init__(config)
        # get Pet tuning model.
        self.pet = pet
        self.pet.pet_config.reg_rules = r'.*dense*|.*linear*|.*mapping*|.*projection*'
        self.backbone = LoraAdapter.get_pet_model(self.backbone, self.pet.pet_config)
        self.load_checkpoint(config)
        # freeze pretrained model
        PetAdapter.freeze_pretrained_model(self, self.pet.pet_type)
```

2. 修改训练任务参数，使用MindFormer的trainer进行模型训练，主要修改模型配置yaml，以GPT2为例：

```yaml
model:
  model_config:
    type: GPT2Config
    ...
  arch:
    # type: GPT2LMHeadModel
    # 替换为适配微调算法的模型
    type: GPT2WithLora
    pet:
      pet_type: lora
      pet_config:
        # configurition of lora
        in_channels: 768
        out_channels: 768
        lora_rank: 8
        lora_alpha: 16
        lora_dropout: 0.1
```

只需要将arch参数由原来的`class GPT2LMHeadModel`替换为现在适配微调算法的`class GPT2WithLora`，并按照lora算法进行微调参数配置。

3. 使用MindFormer的Trainer进行模型训练：

```python
from mindformers.trainer.trainer import Trainer

gpt2_trainer = Trainer(
    task='text_generation',
    model='gpt2',
    pet_method='lora',
    train_dataset="/data/wikitext-2/train",
)

gpt2_trainer.finetune()
```

至此，完成了一个微调算法适配过程，最后执行上述步骤3中的代码即可拉起微调算法的训练流程。
