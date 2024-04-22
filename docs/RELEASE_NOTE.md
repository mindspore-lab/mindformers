# MindSpore Transformers 1.0.2 RELEASE NOTE

以下为MindSpore Transformers (以下称为MindFormers) 套件 1.0.2 版本的变更日志，相较于1.0.1版本有以下关键新特性和bug fix修复。

## 新特性

- [GLM3](./model_cards/glm3.md)/[GLM3-32k](../research/glm32k/glm32k.md)新增支持Paged Attention推理。

## 新模型

以下为 research 模型：

| 模型                                        | 规格          |
|-------------------------------------------|-------------|
| [Qwen1_5](../research/qwen1_5/qwen1_5.md) | qwen1_5_72b |
| [Mengzi3](../research/mengzi3/mengzi3.md) | mengzi3_13b |

## Bugfix

在当前版本发布周期内，我们进行了模型/功能/文档等Bugfix，修复内容如下：

- [#I9EWKI](https://gitee.com/mindspore/mindformers/issues/I9EWKI)：修复了离线推理启动脚本[run_infer_main.py](https://gitee.com/mindspore/mindformers/blob/r1.0/run_infer_main.py)中dynamic开关和paged attention开关同时开启时报错的问题。
- [#I9G6BG](https://gitee.com/mindspore/mindformers/issues/I9G6BG)：修复了多卡权重自动转换Rank 0进程出错时，其他Rank进程不会自动终止的问题。

# MindSpore Transformers 1.0.1 RELEASE NOTE

以下为MindSpore Transformers (以下称为MindFormers) 套件 1.0.1 版本的变更日志，相较于1.0.0版本有以下关键bug fix修复。

## Bugfix

在当前版本发布周期内，我们进行了模型/功能/文档等Bugfix，修复内容如下：

- [#I91T78](https://gitee.com/mindspore/mindformers/issues/I91T78)：修复了大模型断点续训时日志显示的epoch与所加载ckpt的epoch不一致问题。

# MindSpore Transformers 1.0.0 RELEASE NOTE

以下为MindSpore Transformers (以下称为MindFormers) 套件 1.0.0 版本的变更日志

## 新特性

- [LLM数据在线加载](./feature_cards/LLM_DataLoader.md)：对于LLM模型的训练场景，该特性支持直接读取非MindRecord格式的数据，如json、parquet等，减少了将数据转换为MindRecord格式的工作量；
- [Flash Attention](./feature_cards/Training_Algorithms.md#flash-attention)：Flash Attention（简称FA），是深度学习业界主流的注意力计算加速算法；MindSpore+Ascend架构也提供了FA实现，当前MindFormers对部分模型进行了FA的适配，可使用 `model_config` 中的 `use_flash_attention` 配置项控制模型是否使用FA；依赖MindSpore2.2.10及以上版本；
- [断点续训支持Step级别恢复](./feature_cards/Resume_Training.md)：对断点续训特性进行了更新迭代，现在使用断点续训特性时，可以自动跳过已训练的数据，恢复到断点权重对应的step位置继续训练；
- [梯度累积](./feature_cards/Training_Algorithms.md#梯度累积)：梯度累积算法是业界常用的扩大batch_size，解决OOM的一种算法，MindSpore在2.1.1之后的版本中增加了 `mindspore.nn.wrap.cell_wrapper.GradAccumulationCell` 这一梯度累积实现接口，通过拆分MiniBatch的形式实现了梯度累积；MindFormers套件对上述实现接口进行了适配，新增 `gradient_accumulation_steps` 配置项以控制梯度累积步数；限制：梯度累积当前仅支持在半自动并行模式下使用；
- output文件夹路径支持自定义：MindFormers现在支持配置 `output_dir` 以自定义训练权重，切分策略等文件的保存路径；日志文件的保存路径由环境变量 `LOG_MF_PATH` 控制，可在[环境变量使用说明](https://mindformers.readthedocs.io/zh-cn/r1.0/docs/practice/Environment.html)中查看具体信息；
- [自动并行](./feature_cards/Auto_Parallel.md)：自动并行模式让用户可以无需为网络中的每一个算子配置并行策略，即可达到高效并行训练的效果。详情参考MindSpore官网关于[自动并行](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/auto_parallel.html)的说明。当前本特性为实验性特性，仅在llama2模型上提供了自动并行的使用配置；
- [adaptive loss scale](./feature_cards/Training_Algorithms.md#adaptive-loss-scaling)：现有dynamic loss scaling方案使用固定scale window，在FP16或更低精度(8bit浮点格式)混合精度训练训练时，如果选用较大的scale window，存在loss scaling 调整不及时的风险，影响模型收敛性和收敛速度；如果选用较小的scale window，loss scale调整至合适的值时，仍会频繁上调，损失大量训练数据；Adaptive loss scaling方案，通过动态调节scale window，实现自适应调整loss scale，实时将loss scale调整至FP16和8bit浮点格式正常训练所需的合适的值，同时避免损失大量训练数据；
- [LLM大模型通用export接口](./feature_cards/Inference.md#模型导出增量推理为例)：执行MindSpore Lite推理时需导出MindIR文件，本特性提供了适用于LLM大模型的通用export导出接口，用户可使用接口便捷地完成导出功能；
- [动态组网分布式启动方式](./feature_cards/Dynamic_Cluster.md)：MindSpore2.2.0以上版本提供了动态组网的启动方式，可以在不依赖rank table和第三方库的情况下拉起分布式任务；MindFormers在此提供了相应的脚本和使用教程；
- beam search采样：文本生成新增支持beam search后处理采样，调用model.generate()接口时，num_beams入参设置大于1的整数值即可启用beam search采样；当前尚不支持与增量推理，流式推理特性同时使用；
- 模型权重分次加载：MindFormers新增支持了模型权重分次加载的逻辑，适用于低参微调场景，分别加载base权重和lora微调权重；使用方式可参考[configs/README.md](https://gitee.com/mindspore/mindformers/blob/r1.0/configs/README.md)中关于 `load_checkpoint` 参数的介绍。

## 新模型

| 模型                                    | 规格             |
| --------------------------------------- | ---------------- |
| [CodeGeeX2](./model_cards/codegeex2.md) | codegeex2_6b     |
| [CodeLLaMA](./model_cards/codellama.md) | codellama_34b    |
| [GLM2-PTuning](./model_cards/glm2.md)   | glm2_6b_ptuning2 |
| [GLM3](./model_cards/glm3.md)           | glm3_6b          |
| [GPT2](./model_cards/gpt2.md)           | gpt2_13b         |

以下为 research 模型：

| 模型                                                  | 规格                                  |
| ----------------------------------------------------- | ------------------------------------- |
| [InternLM](../research/internlm/internlm.md)          | interlm_20b (仅推理)                  |
| [Qwen](../research/qwen/qwen.md)                      | qwen_7b<br/>qwen_7b_lora<br/>qwen_14b |
| [Skywork](../research/skywork/skywork.md)             | skywork_13b                           |
| [VisualGLM](../esearch/visualglm/visualglm.md)        | visualglm_6b                          |
| [WizardCoder](../research/wizardcoder/wizardcoder.md) | wizardcoder_15b                       |

## Bugfix

在当前版本发布周期内，我们进行了模型/功能/易用性/文档等诸多方面的Bugfix，在此仅列举部分修复内容：

- [#I8URBL](https://gitee.com/mindspore/mindformers/issues/I8URBL)：修复了部分数据集在半自动并行+full_batch下仍错误地进行shard导致的训练数据不正确的问题。
- [#I8JVTM](https://gitee.com/mindspore/mindformers/issues/I8JVTM)：修复了在使用Trainer传入模型实例时，GradAccumulation，PipelineCell等封装工具类未正确生效的问题。
- [#I8L4LZ](https://gitee.com/mindspore/mindformers/issues/I8L4LZ)：修复了断点续训传入数据集实例时无法跳过已训练数据的问题。
- [#I8NHO5](https://gitee.com/mindspore/mindformers/issues/I8NHO5)：修复了get_pet_model方法的加载权重逻辑，解决无法加载部分低参微调模型权重的问题。
- [#I8THC3](https://gitee.com/mindspore/mindformers/issues/I8THC3)：修复了权重切分创建软链接时多进程读写操作冲突的问题。

欢迎对本项目提出意见与建议，以帮助项目持续改进。

## 贡献者

感谢以下人员做出的贡献：

Chenhua Geng, dingxu (E), fushengshi, heqinglin, koukairui, liuzhidan, renyujin, shuchi, Zhenhao Li, ZhidanLiu, 陈心锐, 陈子恒, 冯浩, 胡桂鹏, 胡思超, 黄磊, 黄生帅, 黄欣靓, 黄勇, 黄子灵, 姜海涛, 焦毅, 李兴炜, 林鑫, 倪钰鑫, 彭康, 苏海波, 田凯, 杨贵龙, 杨路航, 余金, 张森镇, 张小雯, 张又文, 赵栢杨, 周胜凯, 朱国栋

欢迎以任何形式对项目提供贡献！
