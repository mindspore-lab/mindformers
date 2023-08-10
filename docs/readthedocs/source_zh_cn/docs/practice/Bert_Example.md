# 开发新模型--以Bert为例

本文将介绍如何在mindformers中构建一个自己的模型。这里将以Bert为例，构建其训练、推理的相关代码。

## 模型注册

在mindformers中，我们主要使用注册机制来方便用户能使用AutoConfig、AutoModel等通用接口进行模型的调用。因此，开发一个全新的模型，我们也希望用户能使用库上的相应接口，对模型进行注册。

以Bert模型为例。首先，我们通常将模型相关的文件放置在`mindformers/models`文件夹中。为了新建立一个模型，我们在`models`路径下，新建了一个`bert`文件夹。在该文件夹中，我们可以放置模型文件、配置文件、处理文件、分词文件等与模型相关的文件。bert模型注册的代码结构如下：

```python
# mindformers/models/bert/bert.py
from mindformers.tools.register import MindFormerRegister
from mindformers.models.base_model import BaseModel

@MindFormerRegister.register(MindFormerModuleType.MODELS)
class BertNetwork(BaseModel):
    _support_list = MindFormerBook.get_model_support_list()['bert']
    def __init__(self, config):
        super(BertNetwork, self).__init__(config)

    def construct(self, **inputs):
        # bert model code
        pass
```

在上述代码中，有三点需要详细介绍：

* bert模型通过`MindFormerRegister`的`register`函数，将bert模型注册到`MindFormerRegister`的`registry`字典中，方便后续的调用。具体注册方式，可以参考[register.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/register/register.py)。

* `BertNetwork`需要继承`BaseModel`，从而继承其自带的方法，包括`from_pretrained`、`save_pretrained`等，可以帮助用户快速从模型名称、路径等方式，生成模型。

* `_support_list`是模型自带的支持列表，其包含了模型所对应的支持列表。用户可以调用`BertNetwork.show_support_list()`方法，获取不同大小的bert模型的支持列表，从而选择自己所需要对应的模型结构。在support_list中，我们将当前的模型命名为"bert_base_uncased"。

具体代码，可以参见[bert.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/bert/bert.py)。

### 配置文件注册

对于`BertNetwork`而言，需要传入一个`config`文件进行模型的配置，其中包含`num_layers`、`embedding_size`等配置信息，当然，这是由用户自己决定需要传给模型什么样的信息的。对于这样的配置信息，我们也提供了相应的注册机制，可以方便用户快速地获取到模型的配置。在`bert`文件夹中新建`bert_config.py`，其配置信息的注册代码如下：

```python
# mindformers/models/bert/bert_config.py
from mindformers.tools.register import MindFormerRegister
from mindformers.models.base_config import BaseConfig

@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class BertConfig(BaseConfig):
    _support_list = MindFormerBook.get_model_support_list()['bert']
    def __init__(self, num_layers: int = 12, **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        self.num_layers = num_layers
```

同样地，在注册类的构造中，我们也需要完成`register`、`_support_list`等相关信息的注册，从而完成后期的调用。由于继承了`BaseConfig`基类，`BertConfig`也继承了其对应的方法，可以方便地进行配置类的生成和保存，详情可以参见[base_config.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/bert/bert_config.py)。

完成配置文件之后，还需要在[`mindformer_book.py`](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/mindformer_book.py)中进行注册。

首先，我们需要在`_MODEL_SUPPORT_LIST`列表中添加对应的模型名称：

```python
_MODEL_SUPPORT_LIST = OrderedDict([
    ('bert', [
            'bert_base_uncased',
        ]),
])
```

注册完成之后，模型即可通过`AutoConfig`即可通过模型名称`bert_base_uncased`来获取其yaml配置文件的路径地址。

### yaml文件配置

为了方便用户管理和修改其配置文件信息，mindformers中提供了读取`yaml`文件进行配置的方法。yaml文件可以放置在：

```text
/configs/bert/model_config
```

其包含的yaml文件的格式如下：

```text
# /configs/bert/model_config/bert_base_uncased.yaml
model:
  model_config:
    type: BertConfig
    num_labels: 1
```

用户只需要将对应的属性填写到`model_config`中，即可完成yaml文件的修改。

具体代码，可以参见[bert_config.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/bert/bert.py)、[bert_base_uncased.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/bert/model_config/bert_base_uncased.yaml)

在完成配置文件和模型文件的构造后，我们就可以直接使用如下代码进行`Bert`模型的构造：

```python
import os
from mindformers.mindformer_book import MindFormerBook
from mindformers import BertNetwork, BertConfig

# yaml_path
yaml_path = os.path.join(MindFormerBook.get_project_path(), "configs",
                         "bert", "model_config", "bert_base_uncased.yaml")

# 先生成config，再生成模型
config = BertConfig.from_pretrained(yaml_path)
model = BertNetwork(config)
print(model)
```

## AutoModel/AutoConfig使用

Mindformers提供了更加泛化的`AutoModel`和`AutoConfig`接口，进行模型及其配置文件的构造。用户在完成上述的模型的构造和配置文件构造后，得益于`MindFormerRegister`, 可以直接使用`AutoModel`和`AutoConfig`调用刚才注册的`BertNetwork`和`BertConfig`进行使用，代码如下：

```python
import os
from mindformers.mindformer_book import MindFormerBook
from mindformers import AutoModel, AutoConfig

# yaml_path
yaml_path = os.path.join(MindFormerBook.get_project_path(), "configs",
                         "bert", "model_config", "bert_base_uncased.yaml")

# 方法一：先生成config，再生成模型
config = AutoConfig.from_pretrained(yaml_path)
model = AutoModel.from_config(config)
print(model)
# 方法二：AutoModel 可以直接通过config的yaml文件进行模型的生成
model = AutoModel.from_config(yaml_path)
print(model)
```

在Mindformers中，我们提供了从云上下载配置的功能。代码如下：

```python
import os
from mindformers.mindformer_book import MindFormerBook
from mindformers import AutoModel, AutoConfig

model_name = "bert_base_uncased"
config = AutoConfig.from_pretrained(model_name)
model = AutoModel.from_config(config)
```

## Trainer构建

构造完成模型和配置文件后，就可以构建代码进行模型的训练、评估、预测等操作。同样地，mindformers提供了`trainer`类，通过`trainer`，调用模型、数据集等。在mindformers中，需要在trainer中构建对应的类，一般我们使用任务名称来建立`trainer`类。

### 模型tranier构建

根据需要构建的模型任务类型，我们可以在`trainer`文件夹下的新建一个`masked_language_modeling`文件夹，在该文件夹下面，我们可以构建对应的任务的文件，如：`masked_language_modeling_pretrain.py`。与上面的配置和模型相同，我们也提供了`BaseTrainer`作为基类，新构建的模型的类可以继承该基类。具体代码如下

```python
from mindspore.train.model import Model

@MindFormerRegister.register(MindFormerModuleType.TRAINER, alias="mlm")
class MaskedLanguageModelingTrainer(BaseTrainer):
    r"""MaskedLanguageModeling Task For Trainer.
    """
    def __init__(self, model_name: str = None):
        super(MaskedLanguageModelingTrainer, self).__init__(model_name)

        def train(self,
              config=None,
              network=None,
              dataset=None,
              wrapper=None,
              optimizer=None,
              callbacks=None,
              **kwargs):
            # your code
            model = Model(wrapper)
            model.train(config.epochs,
                        dataset=dataset,
                        callbacks=callbacks,
                        dataset_sink_mode=config.sink_mode,
                        sink_size=config.sink_mode)

        def eval(self, **kwargs):
            pass

        def predict(self, **kwargs):
            pass
```

在`MaskedLanguageModelingTrainer`中，我们注意到`train`的方法中，我们需要将模型训练所需要的参数都作为入参传入到该函数中，以供模型进行训练。在`train`的方法中，我们可以按MindSpore的训练方法进行训练, 用户可以参考MindSpore的[教程](https://www.mindspore.cn/tutorials/zh-CN/r1.9/advanced/model/model.html)。具体代码可以参考[masked_language_modeling_pretrain.py](https://gitee.com/jiahongqian/mindformers/blob/dev/mindformers/trainer/masked_language_modeling/masked_language_modeling_pretrain.py)。

### 数据集构建

在mindformers中，用户可以在`mindformers/dataset`路径下构建自己的数据集文件，其代码结构如下：

```python
    from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
    from mindformers.base_dataset import BaseDataset

    @MindFormerRegister.register(MindFormerModuleType.DATASET)
    class MaskLanguageModelDataset(BaseDataset):
        def __new__(cls, dataset_config = None):
            # fake dataset generate
            def generator():
                """dataset generator"""
                data = np.random.randint(low=0, high=15, size=(128,)).astype(np.int32)
                input_mask = np.ones_like(data)
                token_type_id = np.zeros_like(data)
                next_sentence_lables = np.array([1]).astype(np.int32)
                masked_lm_positions = np.array([1, 2]).astype(np.int32)
                masked_lm_ids = np.array([1, 2]).astype(np.int32)
                masked_lm_weights = np.ones_like(masked_lm_ids)
                train_data = (data, input_mask, token_type_id, next_sentence_lables,
                            masked_lm_positions, masked_lm_ids, masked_lm_weights)
                for _ in range(256):
                    yield train_data
            # Dataset and operations
            dataset = GeneratorDataset(generator, column_names=["input_ids", "input_mask", "segment_ids",
                                                                "next_sentence_labels", "masked_lm_positions",
                                                                "masked_lm_ids", "masked_lm_weights"])
            dataset = dataset.batch(batch_size=dataset_config.batch_size)
            return dataset
```

对于`Bert`，用户可以参考库上的[`mask_language_model_dataset.py`](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/dataset/mask_language_model_dataset.py)。

### 使用模型trainer进行训练

在完成`trainer`的构建后，用户可以使用如下代码进行模型的训练：

```Python
import os
from mindformers.mindformer_book import MindFormerBook
from mindformers import BertNetwork, BertConfig
from mindformers.trainer import MaskedLanguageModelingTrainer
from mindformers.trainer.config_args import ConfigArguments, RunnerConfig, WrapperConfig
from mindformers.dataset import MaskLanguageModelDataset

# yaml_path
yaml_path = os.path.join(MindFormerBook.get_project_path(), "configs",
                         "bert", "model_config", "bert_base_uncased.yaml")

# 先生成config，再生成模型
config = BertConfig.from_pretrained(yaml_path)
bert_model = BertNetwork(config)

# train config
batch_size = 16
train_config = RunnerConfig(epochs=1, batch_size=batch_size, sink_mode=True, sink_size=2)


# train dataset

dataset = MaskLanguageModelDataset(train_config)

# optimizer
lr_schedule = WarmUpLR(learning_rate=0.001, warmup_steps=100)
optimizer = AdamWeightDecay(beta1=0.009, beta2=0.999,
                            learning_rate=lr_schedule,
                            params=bert_model.trainable_params())

# wrapper
bert_wrapper = TrainOneStepCell(bert_model, optimizer, sens=1024)

# callback
loss_cb = LossMonitor(per_print_times=2)
time_cb = TimeMonitor()
callbacks = [loss_cb, time_cb]

# trainer
bert_trainer = MaskedLanguageModelingTrainer(model_name="bert_base_uncased")

# train model
bert_trainer.train(
              config=train_config,
              network=bert_model,
              dataset=dataset,
              wrapper=bert_wrapper,
              optimizer=optimizer,
              callbacks=callbacks)
```

在这里，我们简单地使用MindSpore自带的`Model`进行模型的训练，在这里，我们需要传入数据集、优化器等组件，这些组件由用户手动定义。为了更好地方便用户管理训练所需要的各个组件，我们提供了一整套任务的配置文件读取，可以让用户快速地获取模型配置、模型训练、数据集配置等所需要的所有信息。

### task config 配置

在建立`trainer`类之后，我们发现，使用模型的`trainer`训练模型，我们需要预先定义好network、dataset、oprimizer等
我们需要一个类来存储模型的训练各项超参数，包括训练的epoch、learning rate、batch size等重要参数。在mindformers中，我们通常使用`yaml`文件进行全局控制。与配置文件类型，一般我们会在`configs`下面的`bert`文件夹构建以下几个文件:

```text
└─configs
  ├─bert
    ├─model_config
        ├─bert_base_uncased.yaml      # 模型的配置文件
    ├─task_config
        ├─context.yaml                # 全局context配置文件
        ├─runner.yaml                 # 运行超参配置文件
        ├─bert_dataset.yaml           # 数据集配置文件
    run_bert_base_uncased.yaml        # 总体配置文件
```

* `bert_base_uncased.yaml`: 模型配置文件，同上述章节所介绍的，提供了模型的配置信息。

* `context.yaml`: 该文件主要提供了全局的参数配置信息，在配置文件中，可以直接使用对应的`context`、`parallel_context`方法中所需要的属性，用户需要了解以下两个MindSpore的函数的相关特性:

    ```python
    from mindspore import context
    context.set_context()
    context.set_auto_parallel_context()
    ```

* `runner.yaml`: 运行配置文件，该文件中存放了运行所需要的runner_config、runner_wrapper、optimizer等信息。其文件格式如下：

    ```text
    # config
    runner_config:
      epochs: 1
      batch_size: 16
      sink_mode: True
      per_epoch_size: -1
      initial_epoch: 0

    # wrapper
    runner_wrapper:
      type: TrainOneStepCell
      sens: 1024

    # optimizer
    optimizer:
      type: AdamWeightDecay
      beta1: 0.9
      beta2: 0.999

    # lr sechdule
    lr_schedule:
      type: WarmUpLR
      learning_rate: 0.0001
      end_learning_rate: 0.000001
      warmup_steps: 10000

    # callbacks
    callbacks:
      - type: MFLossMonitor
      - type: SummaryMonitor
        keep_default_action: True
    ```

* `bert_dataset.yaml`: 在数据集的配置文件中, 我们也可以指定模型的数据集的相关信息。

    ```text
    train_dataset: &train_dataset
        batch_size = 16

    train_dataset_task:
        type: MaskLanguageModelDataset
        dataset_config: *train_dataset
    ```

* `run_bert_base_uncased.yaml`: 该文件会包含上述构建的yaml文件，其格式如下：

    ```text
    base_config: [
        '../__base__.yaml',
        './task_config/context.yaml',
        './task_config/runner.yaml',
        './task_config/bert_dataset.yaml',
        './model_config/bert_tiny_uncased.yaml' ]

    profile: False
    use_parallel: False

    seed: 0
    resume_or_finetune_checkpoint: ''

    run_status: 'train'
    trainer:
      type: mlm
      model_name: 'bert'
    ```

    这里，使用`base_config`将所有的关键配置全部包含，并且使用`run_status`确定模型的运行状态。在`trainer`中，使用`type`关键字，来获取上节构建的`trainer`，并且通过`model_name`获取模型的名称。

    用户可以参考mindformers中已经完成的[Bert配置文件](/home/jenkins/qianjiahong/mindformers/20230113/transformer/configs/bert)进行配置。

### trainer 配置文件注册

同模型的配置文件注册相同, 这里也需要将`trainer`的yaml文件配置信息注册在`mindformer_book.py`中，其注册的位置在：

```python
import os

_TRAINER_SUPPORT_TASKS_LIST = OrderedDict([
    ("fill_mask", OrderedDict([
    ("bert_base_uncased", os.path.join(
        _PROJECT_PATH, "configs/bert/run_bert_base_uncased.yaml")),])
    ),
])

```

这里值得注意的是，`fill_mask`是任务的名称，用户也可以根据任务的特殊性，来命名合适的任务。

## Trainer接口使用

在完成训练任务的配置文件和注册后，用户就可以使用mindformer提供的`Trainer`接口快速调用其网络进行训练，代码如下：

```python
from mindformers.trainer import Trainer

bert_trainer = Trainer(task='fill_mask')
bert_trainer.train()

```

可以看到，相比于`使用模型trainer进行训练`中使用的`MaskedLanguageModelingTrainer`而言，使用`Trainer`及yaml文件进行训练，会更加方便和简洁。

在完成上述训练类的定义后，`MindFormers`也提供了一个统一的运行文件[run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/dev/run_mindformer.py)，方便读者训练自己的网络，运行指令如下：

```shell
python run_mindformer.py --config /path/run_bert_base_uncased.yaml --run_mode train --dataset_dir /datasetpath/
```
