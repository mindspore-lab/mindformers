# 权重离线切分转换

该特性适用于：1.  权重过大，单卡无法加载；2. 权重为切分权重且与目标网络和运行卡数不匹配；

此时可利用本特性进行权重的切分和转换，以适配目标网络运行；

使用场景：1. 分布式恢复训练（恢复时卡数或者并行策略发生改变）；2. 评估/推理场景（权重需要进一步切分或者合并）

## 方案1：源码执行

- step1（默认已有待切分权重相对应的策略文件，若没有，也可参考以下方法生成）

在config中配置`only_save_strategy: True`，正常启动分布式训练/评估/推理，生成目标卡数的分布式策略文件后将会退出。生成的分布式策略文件保存在`output/strategy`目录下。

```yaml
only_save_strategy: True
```

- step2

运行如下脚本完成权重切分转换

```shell
python mindformers/tools/transform_ckpt.py --src_ckpt_strategy SRC_CKPT_STRATEGY --dst_ckpt_strategy DST_CKPT_STRATEGY --src_ckpt_dir SRC_CKPT_DIR --dst_ckpt_dir DST_CKPT_DIR
```

参数说明:

`src_ckpt_strategy`：待转权重的分布式策略文件路径。
  若为None,表示待转权重为完整权重;
  若为切分策略文件,表示原始的权重对应的策略文件;
  若为文件夹,表示需要合并文件夹内策略文件(仅在流水并行生成的策略文件时需要),合并后的策略文件保存在`SRC_CKPT_STRATEGY/merged_ckpt_strategy.ckpt`路径下;

`dst_ckpt_strategy`：目标权重的分布式策略文件路径。即step1中生成的分布式策略文件路径。
  若为None,表示将待转权重合并为完整权重;
  若为切分策略文件,表示目标卡数对应的策略文件
  若为文件夹,表示需要合并文件夹内策略文件(仅在流水并行生成的策略文件时需要),合并后的策略文件保存在`DST_CKPT_STRATEGY/merged_ckpt_strategy.ckpt`路径下;

`src_ckpt_dir`: 待转权重路径，须按照`SRC_CKPT_DIR/rank_{i}/checkpoint_{i}.ckpt`存放，比如单一权重存放格式为`SRC_CKPT_DIR/rank_0/checkpoint_0.ckpt`。

`dst_ckpt_dir`：目标权重保存路径，为自定义空文件夹路径，转换后模型以`DST_CKPT_DIR/rank_{i}/xxx.ckpt`存放。

- step3

将`config`的配置文件中`load_checkpoint`关键字指定为转换的目标权重保存路径，若转换后仍为切分权重，传入转换后的权重文件夹路径即可；若转换后为完整权重，传入权重文件路径即可正常启动训练。

```yaml
load_checkpoint: "{转换后权重文件夹/文件路径}"
```

## 方案2：高阶API 执行

参考Trainer API使用

使用`TrainingArguments`类打开`only_save_strategy`字段，其余步骤可参考**方案1**
