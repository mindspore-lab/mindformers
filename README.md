# transformer

## 介绍

{**以下是 Gitee 平台说明，您可以替换此简介**
Gitee 是 OSCHINA 推出的基于 Git 的代码托管平台（同时支持 SVN）。专为开发者提供稳定、高效、安全的云端软件开发协作平台
无论是个人、团队、或是企业，都能够用 Gitee 实现代码托管、项目管理、协作开发。企业项目请看 [https://gitee.com/enterprises](https://gitee.com/enterprises)}

## 软件架构

```text
tasks: 下游任务
examples:运行脚本
```

## 快速上手

1. 数据预处理：

2. 单卡训练gpt模型

```bash
bash examples/pretrain_gpt.sh  DEVICE_ID EPOCH_SIZE DATA_DIR
```

3. 单机8卡训练gpt模型

```bash
bash examples/pretrain_gpt_distributed.sh 8 hostfile /path/dataset
```
