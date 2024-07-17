# 流水并行的负载偏置与重计算

在大模型训练调优过程中，设备内存的合理使用和分配是一项重要的环节，除了通过各种并行方式将模型、优化器状态等数据切分到不同设备外，还可以通过调整流水并行的负载偏置与重计算来精细调整内存的使用。
在mindformers/models/utils.py中提供了set_layer_stage_recompute函数，用于灵活配置每一层的stage_id与重计算。

## 配置流水并行的负载均衡

流水并行默认网络层数num_layers可以被pp数pipeline_stage整除，每个stage中包含num_layers/pipeline_stage层。
如果网络层数num_layers不能被pp数pipeline_stage整除，或者调整每个stage中包含的层数，那么可以通过offset参数进行配置。

offset可以传入一个list或tuple，此时，list的元素个数需要等于pipeline_stage，list的各元素求和需要等于num_layers % pipeline_stage。
每个stage的层数则为n_layer_i = (num_layer // pipline_stage) + offset[i]

例如，一个网络有48层，pp数为5,offset设为[0,1,1,1,0]，那么这5个stage的层数为9,10,10,10,9。

## 配置重计算与选择重计算

重计算可以显著降低训练时使用的内存，但会额外增加一些计算。

当recompute为True且select_recompute为False时，会对整网配置重计算，此时内存占用最少但性能劣化较多。

当配置选择重计算select_recompute开启时，recompute配置不再生效，可以配置选择重计算select_recompute与选择通信重计算select_comm_recompute。

- **select_recompute**: 选择重计算,会重计算FFN中的Mul与Silu算子，以及在不开FlashAttention时重计算SelfAttention。
- **select_comm_recompute**:选择通信重计算,开启后重计算attention_norm与ffn_norm的norm算子的通信，仅在开启细粒度多副本fine_grain_interleave时使用。
- **parallel_optimizer_comm_recompute**:优化器并行通信重计算,开启后在自动并行或半自动并行模式下，指定Cell内部由优化器并行引入的AllGather通信是否重计算。
- **mp_comm_recompute**:模型并行通信重计算,开启后在自动并行或半自动并行模式下，指定Cell内部由模型并行引入的通信操作是否重计算。
- **recompute_slice_activation**:切片重计算,是否对将保留在内存中的Cell输出进行切片

此外，选择重计算select_recompute与选择通信重计算select_comm_recompute除了可以配置True或False外，还可以传入list来具体指定每个stage中有多少层开启对应的选择重计算，list元素个数需要等于pipeline_stage个数。
例如：

```yaml
# recompute config
recompute_config:
  recompute: False
  select_recompute: [5, 5, 5]
  select_comm_recompute: [5, 5, 5]
  parallel_optimizer_comm_recompute: False
  mp_comm_recompute: True
  recompute_slice_activation: True
```
