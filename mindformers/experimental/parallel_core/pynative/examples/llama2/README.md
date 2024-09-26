# 训练 Llama2

## 预训练 Llama2 模型

### 准备工作

#### 下载 llama2 权重

请下载好 llama2 权重

#### 数据准备

运行下面命令，准备好数据集.该命令会下载，解压，清洗和 tokenize`wikitext-2-v1`数据集，最终结果会储存在`dataset`目录下

```bash
bash prepare_dataset_wiki_pretrain_llama2.sh dataset {path_to_your_llama_ckpt}
```

其中:

- dataset 是目标文件夹名
- {path_to_your_llama_ckpt}是你的 llama ckpt 路径，其中需要包含`tokenizer.model`文件

### 预训练

1.修改配置文件

训练和模型所有的相关配置都用一个 yaml 文件来管理，你可以修改`train_llama.yaml`文件来配置训练参数，具体参数可以参考[config 类代码](../distri_cores/config.py)。

2.将`train.py`里的`CONFIG_PATH`来指定你的配置文件路径。

3.运行训练脚本：

```bash
bash train_llama2.sh {config_path}
```

其中:

- {config_path}是你的配置文件路径, 默认为`pretrain_llama2.yaml`
- 你可以修改`train_llama.sh`中的`ASCEND_RT_VISIBLE_DEVICES`来指定使用的设备。log 文件会保存在`./msrun_log`目录下。
