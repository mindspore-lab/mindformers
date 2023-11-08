# CLIP

## 模型描述

CLIP (Contrastive Lanuguage-Image Pre-Training)：是一种基于图文对进行训练的transformer模型，在预训练完成以后，任意给定一张图片，它可以在不用微调的情况下，完成对图片的零样本分类。

[论文](https://arxiv.org/abs/2103.00020) Alec Radford, Jong Wook Kim, et al., Learning Transferable Visual Models From Natural Language Supervision, 2021.

注：CLIP训练代码未开源，故MindFormers提供训练pretrain、finetune功能，但不不保证精度，目前仅对zero shot图片分类精度做了对齐。

## 数据集准备

### 预训练使用数据集：Flickr8k([链接](https://pan.baidu.com/s/1LRlQUL1MRipPL4MLOdExzg)，密码: s4be)

- 数据集大小：2.2G，共8000张彩色图像，每张图像都与五个不同的标题配对，这些标题提供了对图片中物体和事件的内容描述
    - 训练集：6000张图像
    - 验证集：1000张图像
    - 测试集：1000张图像
- 数据格式：RGB

 ```bash
数据集目录格式
└─Flickr8k
    ├─Flickr8k_Dataset
    |      └─Flickr8k_Dataset
    └─Flickr8k_text
           ├─Flickr8k.devImages.txt
           ├─Flickr8k.testImages.txt
           ├─Flickr8k.trainImages.txt
           └─Flickr8k.token.txt
 ```

### 零样本下游任务使用的数据集：[Cifar100](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

- 数据集大小：161M，共60000张图片，100个类别
    - 训练集：50000张图片
    - 测试集：10000张图片
- 数据格式：二进制文件

 ```bash
数据集目录格式
└─cifar-100-python
    ├─meta
    ├─test  
    └─train  
 ```

## 快速使用

### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](../../README.md#方式一使用已有脚本启动)

- 脚本运行测试

当前clip多卡精度有异常，仅支持单卡，后续版本会修复

```shell
# pretrain
python run_mindformer.py --config ./configs/clip/run_clip_vit_b_32_pretrain_flickr8k.yaml --run_mode train --train_dataset_dir [DATASET_PATH]

# evaluate
python run_mindformer.py --config ./configs/clip/run_clip_vit_b_32_zero_shot_image_classification_cifar100.yaml --run_mode eval --eval_dataset_dir [DATASET_PATH]

# predict
python run_mindformer.py --config ./configs/clip/run_clip_vit_b_32_zero_shot_image_classification_cifar100.yaml --run_mode predict --predict_data [PATH_TO_IMAGE]
```

### 调用API启动

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

- Model调用接口

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers import CLIPModel, CLIPConfig

CLIPModel.show_support_list()
# 输出：
# - support list of CLIPModel is:
# -    ['clip_vit_b_32', 'clip_vit_B_16', 'clip_vit_l_14', 'clip_vit_l_14@336']
# - -------------------------------------

# 模型标志加载模型
model = CLIPModel.from_pretrained("clip_vit_b_32")

#模型配置加载模型
config = CLIPConfig.from_pretrained("clip_vit_b_32")
# {'text_config': {'hidden_size': 512, 'vocab_size': 49408, 'max_position_embeddings': 77,
# 'num_hidden_layers': 12}, 'vision_config': {'hidden_size': 768, 'image_size': 224, 'patch_size': 32,
# 'num_hidden_layers': 12}, 'projection_dim': 512, 'ratio': 64, 'checkpoint_name_or_path': 'clip_vit_b_32',
# 'dtype': 'float16'}
model = CLIPModel(config)
```

- Trainer接口开启训练/评估/推理：

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers.trainer import Trainer
from mindformers.tools.image_tools import load_image
# 初始化预训练任务
trainer = Trainer(task='contrastive_language_image_pretrain',
    model='clip_vit_b_32',
    train_dataset='./Flickr8k')
trainer.train() # 开启预训练

#初始化零样本图像分类下游任务
trainer = Trainer(task='zero_shot_image_classification',
    model='clip_vit_b_32',
    eval_dataset='./cifar-100-python')  
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

# 方式1: 使用训练好的权重进行评估和推理
trainer.evaluate(eval_checkpoint=True)
predict_result = trainer.predict(predict_checkpoint=True, input_data=img, top_k=3)
print(predict_result)

# 方式2: 从obs下载训练好的权重并进行评估和推理
trainer.evaluate()  #下载权重进行评估
predict_result = trainer.predict(input_data=img, top_k=3)  #下载权重进行推理
print(predict_result)
```

- pipeline接口开启快速推理

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers import pipeline
from mindformers.tools.image_tools import load_image

classifier = pipeline("zero_shot_image_classification",
                      model="clip_vit_b_32",
                      candidate_labels=["sunflower", "tree", "dog", "cat", "toy"])
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
          "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
classifier(img)
# 输出
# [[{'score': 0.99995565, 'label': 'sunflower'}, {'score': 2.5318595e-05, 'label': 'toy'},
# {'score': 9.903885e-06, 'label': 'dog'}, {'score': 6.75336e-06, 'label': 'tree'},
# {'score': 2.396818e-06, 'label': 'cat'}]]
```

## 模型性能

| model |           task_type            |                                  model_Type                                   | datasets |              Top1-accuracy              | log |                                                                                                example                                                                                                |
|:-----:|:------------------------------:|:-----------------------------------------------------------------------------:|:--------:|:---------------------------------------:|:---:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| clip  |           pretrained           | clip_vit_b_32 <br> clip_vit_b_16 <br> clip_vit_l_14 <br> clip_vit_l_14@336 | flickr8k |                    \                    |  \  |                                               pretrain [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/contrastive_language_image_pretrain/clip_vit_b_32_pretrain_on_flickr8k.sh)                                               | \|
| clip  | zero_shot_image_classification | clip_vit_b_32 <br> clip_vit_b_16 <br> clip_vit_l_14 <br> clip_vit_l_14@336 | cifar100 | 57.24%<br>61.41%<br>69.67%<br>68.19% |  \  | eval [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/zero_shot_image_classification/clip_vit_b_32_eval_on_cifar100.sh) <br> predict [link](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/zero_shot_image_classification/clip_vit_b_32_predict_on_cifar100.sh) |

## 模型权重

本仓库中的`clip_vit_b_32`来自于openai/clip的[`ViT-B/32`](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt), 基于下述的步骤获取：

1. 从上述的链接中下载`ViT-B/32`的模型权重

2. 执行转换脚本，得到转换后的输出文件`clip_vit_b_32.ckpt`

其余参数获取方式相同

```shell
python mindformers/models/clip/convert_weight.py --torch_path "PATH OF ViT-B/32.pt" --mindspore_path "SAVE PATH OF clip_vit_b_32.ckpt"
```