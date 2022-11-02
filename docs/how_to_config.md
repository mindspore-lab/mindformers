# 使用指南

## 自定义参数

目前参数的传入主要采用传入`yaml`文件+命令行参数覆盖的方式。例如下文所示

```bash
python -m transformer.trainer.trainer \
    --auto_model="gpt" \
    --epoch_size=$EPOCH_SIZE \
    --train_data_path=$DATA_DIR \
    --optimizer="adam"
    --custom_args="test" \
```

`config`作为命令行解析的第一个参数，将从指定的文件中加载所有的参数。然后开始解析其后面的
参数`epoch_size`、`train_data_path`、`optimizer`和`custom_args`等参数。
由于前三个参数已经在`gpt_base.yaml`文件中定义，所以这些参数会被命令行中传入的参数覆盖。

而`custom_args`中没有在配置文件中定义，会被添加到解析的参数中去。用户可以在`train.py`中通过`opt.custom_args`获取。
其中`opt`是`run_train`的入参。

## 添加自定义模型

用户可以在`transformer/models`目录下创建自己的模型文件夹。构建好模型代码后，需要在`tranformer/models/build_model.py`中加入自己模型的
构建接口。

## 添加自定义数据集

用户可以在`transformer/data`目录下创建自己的数据集处理文件。然后在`tranformer/data/build_dataset.py`中加入数据集的构建接口。
构建接口。

## 运行模式

目前脚本根据传入的`parallel_mode`参数来决定运行的模式。目前`parallel_mode`的可选入参为如下:

### 单卡运行

`stand_alone`: 单卡模式。示例脚本可以参考`examples/pretrain_gpt.sh`。此时`parallel_config`中的参数并不会生效。

### 数据并行

`data_parallel`: 数据并行模式。示例脚本可以参考`examples/pretrain_gpt_distributed.sh`。用户需要手动修改`--parallel_mode=data_parallel`
此时`parallel_config`中的参数并不会生效。

### 自动并行模式：

`semi_auto_parall`: 半自动并行模式。此模式下可以使能目前MindSpore提供的所有并行能力。
模型将根据传入的`parallel_config`中配置的模型并行数目对权重进行切分。
用户可以根据自己的需要，在`parallel_mode`为`semi_auto_parall`的模式下，逐步开启如下的并行配置。

目前支持的并行策略如下：

- 数据并行
- 模型并行
- 优化器并行
- 多副本并行

#### 自动并行下的数据并行

>--parallel_mode=semi_auto_parallel --data_parallel=总卡数

用户需要在启动脚本中增加参数。
其中`data_parallel`表示数据并行度，默认值在`gpt_base.yaml`的配置文件中给定，值为1。
此参数下和`--parallel_mode=data_parallel`的区别如下：

- ReduceSum、ReduceMean等操作在`axis`轴进行聚合时，其结果等价在所有卡的输入数据在单卡上的运行结果。

#### 优化器并行

>--parallel_mode=semi_auto_parallel --data_parallel=总卡数 --optimizer_shard=True

用户可以在启动脚本中增加入参来使能此配置。
模型的参数、优化器状态将会进一步在数据并行维度上进行切分，将进一步减少模型参数在每卡的占用。
在使能此项配置后，每卡保存的模型权重是整体的一个切片。

#### 模型并行

>--parallel_mode=semi_auto_parallel --data_parallel=4 --model_parallel=2

当用户需要对模型中的权重进行切分，以进一步减少模型在每卡中占用的内存时，可以增加上述入参。
此时模型中的所有权重将会被切分为`model_parallel`份数。用户需要确保`data_parallel`和`model_parallel`
的乘积等于总卡数。**注意**，由于模型并行会在前向计算和反向计算中引入额外的通信。
推荐的`model_parallel`可选值为2/4/8，此时将确保模型并行产生的通信在单机内。

### 开启重计算

用户可以在启动脚本中增加如下参数开启重计算。开启后，程序能够运行更大的Batch Size或者更大的模型，但是代价是增加更多的计算时间。

>--recompute=True