### AutoClass 组件

#### AutoClass 设计

MindFormers大模型套件提供了AutoClass类，包含AutoConfig、AutoModel、AutoTokenizer、AutoProcessor4个便捷高阶接口，方便用户调用套件中已封装的API接口，上述4类分别提供了相应领域模型的ModelConfig、Model、Tokenzier、Processor的实例化功能。

![输入图片说明](https://foruda.gitee.com/images/1673434276426093311/70cb1623_9324149.png "image-20230104100951903.png")

#### AutoClass

|  AutoClass类  | from_pretrained属性（实例化功能） | from_config属性（实例化功能） |
| :-----------: | :-------------------------------: | :---------------------------: |
|  AutoConfig   |                 √                 |               ×               |
|   AutoModel   |                 √                 |               √               |
| AutoProcessor |                 √                 |               ×               |
| AutoTokenizer |                 √                 |               ×               |

* AutoClass接口代码：[AutoClass](https://gitee.com/mindspore/mindformers/blob/r0.3/mindformers/auto_class.py)
* AutoConfig 使用样例：利用`from_pretrained`属性完成模型配置的实例化

```python
from mindformers.auto_class import AutoConfig
# 1)  instantiates a config by yaml model name
config_a = AutoConfig.from_pretrained('clip_vit_b_32')

# 2)  instantiates a config by yaml model path
from mindformers.mindformer_book import MindFormerBook
config_path = os.path.join(MindFormerBook.get_project_path(),
                           'configs', 'clip', 'model_config', "clip_vit_b_32.yaml")
config_b = AutoConfig.from_pretrained(config_path)
```

* AutoModel使用样例：利用`from_pretrained`或者`from_config`属性完成网络模型的实例化

```python
from mindformers.auto_class import AutoModel
# 1)  input model name, load model and weights
model_a = AutoModel.from_pretrained('clip_vit_b_32')

# 2)  input model directory, load model and weights
from mindformers.mindformer_book import MindFormerBook
checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(), 'clip')
model_b = AutoModel.from_pretrained(checkpoint_dir)

# 3)  input yaml path, load model without weights
config_path = os.path.join(MindFormerBook.get_project_path(),
                           'configs', 'clip', 'model_config', "clip_vit_b_32.yaml")
model_c = AutoModel.from_config(config_path)

# 4)  input config, load model without weights
config = AutoConfig.from_pretrained('clip_vit_b_32')
model_d = AutoModel.from_config(config)
```

* AutoProcessor使用样例：利用`from_pretrained`属性完成数据预处理的实例化

```python
from mindformers.auto_class import AutoProcessor
# 1)  instantiates a processor by yaml model name
pro_a = AutoProcessor.from_pretrained('clip_vit_b_32')

# 2)  instantiates a processor by yaml model path
from mindformers.mindformer_book import MindFormerBook
config_path = os.path.join(MindFormerBook.get_project_path(),
                           'configs', 'clip', 'model_config', "clip_vit_b_32.yaml")
pro_b = AutoProcessor.from_pretrained(config_path)
```

* AutoTokenizer使用样例：利用`from_pretrained`属性完成tokenizer的实例化

```python
from mindformers.auto_class import AutoTokenizer
# 1)  instantiates a tokenizer by the model name
tokenizer_a = AutoTokenizer.from_pretrained("clip_vit_b_32")

# 2)  instantiates a tokenizer by the path to the downloaded files.
from mindformers.models.clip.clip_tokenizer import ClipTokenizer
clip_tokenizer = ClipTokenizer.from_pretrained("clip_vit_b_32")
clip_tokenizer.save_pretrained(path_saved)
restore_tokenizer = AutoTokenizer.from_pretrained(path_saved)
```
