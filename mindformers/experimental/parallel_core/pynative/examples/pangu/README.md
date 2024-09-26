# Pangu Demo

## Usage

### 数据准备

运行下面命令，准备好数据集，该命令会下载，解压，预处理`wikitext-2-v1`数据集，最终结果会储存在`dataset`目录下

```bash
bash prepare_dataset.sh dataset 1025
```

其中:

- dataset 是目标文件夹名
- 1025 是目标序列长度+1
  注意这里使用的是`gpt` tokenizer，pangu 还可以使用`jieba` tokenizer，具体可见[文档](https://gitee.com/mindspore/models/tree/master/official/nlp/Pangu_alpha)。

### 训练

1.修改配置文件

训练和模型所有的相关配置都用一个 yaml 文件来管理，你可以修改`train_pangu.yaml`文件来配置训练参数，具体参数可以参考[config 类代码](../distri_cores/config.py)。

2.将`train.py`里的`CONFIG_PATH`来指定你的配置文件路径。

3.运行`run.sh`脚本：

```bash
bash run.sh
```

你可以修改`run.sh`中的`ASCEND_RT_VISIBLE_DEVICES`来指定使用的设备。log 文件会保存在`./msrun_log`目录下。
