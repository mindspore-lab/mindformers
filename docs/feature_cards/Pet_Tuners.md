# 低参微调

MindPet（Pet：Parameter-Efficient Tuning）是属于Mindspore领域的微调算法套件。随着计算算力不断增加，大模型无限的潜力也被挖掘出来。但随之在应用和训练上带来了巨大的花销，导致商业落地困难。因此，出现一种新的参数高效（parameter-efficient）算法，与标准的全参数微调相比，这些算法仅需要微调小部分参数，可以大大降低计算和存储成本，同时可媲美全参微调的性能。

目前低参微调针对MindFormers仓库已有的大模型进行统一架构设计，对于LLM类语言模型，我们可以统一调度修改，做到只需要调用接口或者是自定义相关配置文件，即可完成对LLM类模型的低参微调算法的适配。

## [微调支持列表](../model_support_list.md#微调支持列表)

## Lora使用示例

1. 确定需要替换的模块，lora模块一般替换transformers模块的query，key，value等线性层，替换时需要找到（query, key, value）等模块的变量量，在统一框架中采用的是正则匹配规则对需要替换的模块进行lora微调算法的替换。

```python
# 以GPT为例，在GPT的attention定义中，我们可以查看到qkv的定义如下：
class MultiHeadAttention(Cell):
    ...
    # Query
    self.dense1 = Linear(hidden_size,
                          hidden_size,
                          compute_dtype=compute_dtype,
                          param_init_type=param_init_type)
    # Key
    self.dense2 = Linear(hidden_size,
                          hidden_size,
                          compute_dtype=compute_dtype,
                          param_init_type=param_init_type)
    # Value
    self.dense3 = Linear(hidden_size,
                          hidden_size,
                          compute_dtype=compute_dtype,
                          param_init_type=param_init_type)
    ...
```

找到如上定义后，在步骤2中则可以定义lora的正则匹配规则为：`r'.*dense1|.*dense2|.*dense3'`

2. 定义lora的配置参数修改已有的配置文件，如根据`configs/gpt2/run_gpt2.yaml`，在`model_config`中增加lora相关的配置，如下所示：

```yaml
model:
  model_config:
    type: GPT2Config
    ...
    pet_config: # configurition of lora
      pet_type: lora
      lora_rank: 8
      lora_alpha: 16
      lora_dropout: 0.1
      target_modules: ".*dense1|.*dense2|.*dense3"
  arch:
    type: GPT2LMHeadModel
```

修改完毕后，可以参考训练流程使用该配置文件进行模型训练。

3. 使用MindFormer的Trainer进行模型训练：

```python
import mindspore as ms
from mindformers.trainer.trainer import Trainer

ms.set_context(mode=0) # 设定为图模式加速

gpt2_trainer = Trainer(
    task='text_generation',
    model='gpt2',
    pet_method='lora',
    train_dataset="/data/wikitext-2/train",
)

gpt2_trainer.finetune()
```

至此，完成了一个微调算法适配过程，最后执行上述步骤3中的代码即可拉起微调算法的训练流程。

## P-Tuning v2使用示例

1. 修改任务模型，以GLM2为例，主要有以下步骤：
    - 继承`ChatGLM2ForConditionalGeneration`
    - 初始化好`PrefixEncoder`
    - 导入预训练权重以及冻结预训练模型权重
    - 为了使用MindFormer的训练流程，将`ChatGLM2WithPtuning2`微调模型注册到MindFormer中
    - `construct`中构造提示向量输入到模型中

具体代码如下：

```python
@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ChatGLM2WithPtuning2(ChatGLM2ForConditionalGeneration):
    def __init__(self, config, **kwargs):
        ckpt_cfg = config.checkpoint_name_or_path
        config.checkpoint_name_or_path = None
        config.pre_seq_len = config.pet_config.pre_seq_len

        super().__init__(config, **kwargs)

        config.pet_config.num_layers = config.num_layers
        config.pet_config.kv_channels = config.kv_channels
        config.pet_config.num_heads = config.num_attention_heads

        self.prefix_encoder = PrefixEncoder(
            config.pet_config.pre_seq_len,
            config.pet_config.num_layers,
            config.pet_config.num_heads,
            config.pet_config.kv_channels,
            config.pet_config.prefix_projection,
            config.pet_config.projection_dim,
            config.pet_config.dropout_prob
        )

        if ckpt_cfg:
            config.checkpoint_name_or_path = ckpt_cfg
            self.load_checkpoint(config)

        PetAdapter.freeze_pretrained_model(self, config.pet_config.pet_type)

    def construct(self, input_ids, ...):
        if not self.use_past or self.is_first_iteration:
            batch_size = input_ids.shape[0]
            prefix_key_values = self.prefix_encoder(batch_size)

        return super().construct(input_ids, ..., prefix_key_values)
```

2. 修改训练任务参数，主要修改模型配置yaml：

```yaml
model:
  model_config:
    type: ChatGLM2Config
    ...
    num_layers: 28
    kv_channels: 128
    num_attention_heads: 32

    pet_config:
      # p-tuning-v2 微调配置
      pet_type: ptuning2
      pre_seq_len: 128
      prefix_projection: False
      projection_dim: 128
      dropout_prob: 0.0
  arch:
    # 替换为适配微调算法的模型
    type: ChatGLM2WithPtuning2
```

3. 为模型每层分别传入提示向量`prefix_key_value`：

```python
class ChatGLM2Transformer(nn.Cell):
    def construct(self, ..., prefix_key_values=None):
        ...
        for i in range(self.num_layers):
            prefix_key_value = None
            if prefix_key_values is not None:
                prefix_key_value = prefix_key_values[i]
            layer = self.layers[i]

            hidden_states = layer(..., prefix_key_value=prefix_key_value)
        ...
```

4. 模型每层Attention计算前调用`Ptuning2Adapter.add_prefix`添加提示向量并刷新`attention_mask`：

```python
class ChatGLM2SelfAttention(nn.Cell):
    def construct(self, ..., prefix_key_values=None):
        ...
        key_layer, value_layer, attention_mask = self.add_prefix_if_need(
            prefix_key_value,
            key_layer,
            value_layer,
            attention_mask
        )
        ...
        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
        ...

    def add_prefix_if_need(self, prefix_key_value, key_layer, value_layer, attention_mask):
        if not isinstance(self.pre_seq_len, int) or self.pre_seq_len <= 0:
            return key_layer, value_layer, attention_mask

        seq_len = key_layer.shape[2]

        key_layer, value_layer = Ptuning2Adapter.add_prefix(
            prefix_key_value,
            key_layer,
            value_layer
        )

        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            prefix_mask = attention_mask.new_zeros((batch_size, 1, seq_len, self.pre_seq_len))
            m_cat = P.Concat(3)
            # [bs, 1, seq_len, pre_seq_len + seq_len]
            attention_mask = m_cat((prefix_mask, attention_mask))

        return key_layer, value_layer, attention_mask
```

5. 适配增量推理：
    - 适配增量推理有效长度变量`batch_valid_length`和`range`，加上提示向量的长度
    - 适配`key_past`、`value_past`初始`shape`，加上提示向量的长度

```python
class ChatGLM2Transformer(nn.Cell):
    def construct(self, ..., batch_valid_length=None, prefix_key_values=None):
        if batch_valid_length is not None and isinstance(self.pre_seq_len, int):
            batch_valid_length = batch_valid_length + self.pre_seq_len
        ...

class ChatGLM2SelfAttention(nn.Cell):
    def __init__(self, config: ChatGLM2Config, layer_number):
        ...
        if self.use_past:
            total_seq_length = self.seq_length
            if isinstance(config.pre_seq_len, int):
                total_seq_length = total_seq_length + config.pre_seq_len
            seq_range = np.arange(total_seq_length).reshape(1, 1, -1)
            self.range = Tensor(
                np.tile(seq_range, (self.batch_size, 1, 1)), mstype.int32)
            ...

class ChatGLM2Block(nn.Cell):
    def __init__(self, config: ChatGLM2Config, layer_number: int):
        ...
        if self.use_past:
            ...
            total_seq_length = self.seq_length
            if isinstance(config.pre_seq_len, int):
                total_seq_length = total_seq_length + config.pre_seq_len

            kv_shape = (config.batch_size, kv_num_partition, total_seq_length, size_per_head)

            self.key_past = Parameter(
                Tensor(np.zeros(shape=kv_shape), self.params_dtype), name="key_past")
            self.value_past = Parameter(
                Tensor(np.zeros(shape=kv_shape), self.params_dtype), name="value_past")
```

6. 使用MindFormer的Trainer进行模型训练：

```python
from mindformers.trainer.trainer import Trainer

trainer = Trainer(
    task='text_generation',
    model='glm2_6b',
    pet_method='ptuning2',
    train_dataset="/path/to/AdvertiseGen/train.json",
)

trainer.finetune(finetune_checkpoint="glm2_6b")
```

至此，完成了一个P-Tuning v2微调算法适配过程。
