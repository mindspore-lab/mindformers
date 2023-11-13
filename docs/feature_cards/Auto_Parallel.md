# 自动并行

自动并行模式让用户可以无需为网络中的每一个算子配置并行策略，即可达到高效并行训练的效果。当前MindSpore支持如下两种不同的自动并行方案：

- [切分策略传播算法](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/sharding_propagation.html)：由少量使用shard接口配置并行策略的算子，向未配置的算子传播并行策略。在传播时，算法会尽量选取最少引发张量重排布通信的策略。
- [双递归策略搜索算法](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/sapp.html)：用户无需进行任何算子并行策略配置。其基于符号运算的代价模型可以自由适配不同的加速器集群，对于巨大网络以及大规模多卡切分能够保证快速生成最优策略。

详情参考官网关于[自动并行](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/auto_parallel.html)的说明。

## 使用说明

**当前本特性为实验性特性**
当前mindformers仓支持使用双递归策略搜索算法，进行自动化的并行策略生成。后续会新增支持切分策略传播算法。目前支持的模型如下所示：

|  模型  |    自动并行算法    | 模型与参数量                          |
| :----: | :----------------: | :------------------------------------ |
| LLaMA2 | 双递归策略搜索算法 | LLaMA2-7B<br>LLaMA2-13B<br>LLaMA2-70B |
| PanGu  | 双递归策略搜索算法 | 支持中                                |

在模型对应的configs路径下，提供了auto parallel的yaml配置文件，用户可以快速启动单机多卡或多机多卡的自动并行训练。

自动并行模式在**mindspore r2.2版本及之后版本**生效，在**mindspore 2.1及之前版本**采用半自动并行模式。用户根据使用mindspore版本选择对应的yaml配置文件。自动并行模式配置文件带有"auto_parallel"后缀。

## batch_size

自动并行下的*batch_size*配置项和半自动并行下的*batch_size*略有不同。这是因为自动并行下不存在半自动并行的数据并行数（*data_parallel_num*）的概念，每个算子的数据并行和模型切分都不尽相同，没有统一的数据并行数和模型并行数。

半自动并行下的*batch_size*代表单卡上实际执行的*batch size*大小，而整个集群执行的*global batch size*公式为：
$$
global\_batch\_size = batch\_size \times data\_parallel\_num \times micro\_batch
$$
自动并行下的*batch_size*代表单个stage内实际执行的*batch size*大小，而整个集群执行的*global batch size*公式为：
$$
global\_batch\_size = batch\_size \times micro\_batch
$$
可以看到自动并行下的*batch_size*实际上相当于半自动并行概念下的 `batch_size * data_parallel_num`

## mem_coeff

在 yaml 配置文件下新增了 mem_coeff 配置项，用来控制自动并行策略生成时，更倾向于数据并行或者模型并行。此配置项的默认值为0.25，更倾向于进行数据并行，但当模型参数量较大时，采用更多的数据并行会更可能出现内存不足的报错。此时，用户可以通过增大 mem_coeff 的值来控制自动并行策略生成更倾向于模型并行，mem_coeff 值越大，模型并行数越大（**建议用户以4为因数倍增**）。

**当前yaml配置文件下的mem_coeff配置值已经是最优，通常不需要用户进行调整。**
