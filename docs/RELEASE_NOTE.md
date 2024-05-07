# MindSpore Transformers 1.1.0 RELEASE NOTE

以下为MindSpore Transformers (以下称为MindFormers) 套件 1.1.0 版本的变更日志，相较于1.0.2版本有以下关键新特性和bug
fix。

## MindSpore版本适配

该版本对MindSpore2.3版本进行了适配，该版本支持MindSpore版本为MindSpore 2.3.0-rc2。

## 新特性

- [msrun启动方式](https://gitee.com/mindspore/mindformers/blob/r1.1.0/README.md#方式一使用msrun方式启动仅适用于配套mindspore23以上版本):
  msrun是动态组网启动方式的封装，用户可使用msrun以单个命令行指令的方式在各节点拉起多进程分布式任务，并且无需手动设置动态组网环境变量，并且无需依赖第三方库以及配置文件。
- [LoRA权重合并](https://gitee.com/mindspore/mindformers/tree/r1.1.0/docs/feature_cards/Transform_Lorackpt.md):
  LoRA权重合并将LoRA分支权重合并到原模型对应权重，合并后权重可以使用原模型直接进行推理。
- [生成任务min_length控制](https://gitee.com/mindspore/mindformers/pulls/2267):
  生成任务支持最短生成长度min_length和最短生成tokens数min_new_tokens配置，用以控制最短生成长度，防止模型生成长度过短。
- [ckpt权重转换至torch bin权重](https://gitee.com/mindspore/mindformers/tree/r1.1.0/docs/feature_cards/Convert_Weight.md):
  使用Mindformers训练得到的ckpt权重，可以通过提供的权重转换功能转换成torch
  bin权重，用于推理评估等下游任务。
- [GLM3支持多轮对话训练](https://gitee.com/mindspore/mindformers/tree/r1.1.0/docs/model_cards/glm3.md#多轮对话格式数据集):
  GLM3模型新增多轮对话训练，提供多轮对话的数据集处理方式。
- **训推一体**: 训推一体通过使用高性能算子库，在MindFormers框架中下发性能优化、tiling
  cache、动态shape、PagedAttention等方式，以在线推理方式达成高效的推理性能，实现训练到推理零成本迁移。目前语言类模型均支持训推一体。
- **BF16训练**：支持模型包含Llama2_7b、Llama2_13b、Llama2_70b、wizardcoder、glm3_6b、qwen_7b、qwen_14b等。
- [学习率优化](https://gitee.com/mindspore/mindformers/pulls/2301)：
  新增学习率CosineAnnealingLR和CosineAnnealingWarmRestarts，及对存量学习率warmup steps及decay
  steps配置，详见[!2300](https://gitee.com/mindspore/mindformers/pulls/2300)。
- [qwen系列支持8k序列长度训练](https://gitee.com/mindspore/mindformers/blob/r1.1.0/research/qwen/qwen.md#全参微调):
  qwen系列支持8k序列长度进行全参微调，支持规模为7b及14b。

## 新模型

以下为 research 模型：

| 模型                                                                                            | 规格            |
|-----------------------------------------------------------------------------------------------|---------------|
| [deepseek](https://gitee.com/mindspore/mindformers/tree/r1.1.0/research/deepseek/deepseek.md) | deepseek_33b  |
| [Llama3](https://gitee.com/mindspore/mindformers/tree/r1.1.0/research/llama3/llama3.md)       | llama3_8b     |
| [mixtral](https://gitee.com/mindspore/mindformers/tree/r1.1.0/research/mixtral/mixtral.md)    | mixtral_8x7b  |
| [qwen_1.5](https://gitee.com/mindspore/mindformers/tree/r1.1.0/research/qwen1_5/qwen1_5.md)   | qwen1.5_72b   |
| [yi](https://gitee.com/mindspore/mindformers/tree/r1.1.0/research/yi/yi.md)                   | yi_6b, yi_34b |

## Bugfix

在当前版本发布周期内，我们进行了模型/功能/易用性/文档等诸多方面的Bugfix，在此仅列举部分修复内容：

- [!2918](https://gitee.com/mindspore/mindformers/pulls/2918):
  修复training_dataloader中在开启isAlign时数组index问题，修复多进程下host内存占用过多问题。
- [!2360](https://gitee.com/mindspore/mindformers/pulls/2360): 修复CrossEntropy损失函数在logits数值较大时计算结果不对问题。
- [#I9BETP](https://gitee.com/mindspore/mindformers/issues/I9BETP)：修复PolynomialWithWarmUpLR学习率与PyTorch实现不一致问题。

## 贡献者

感谢以下人员做出的贡献：

Chenhua Geng, dingxu (E), heqinglin, koukairui, renyujin, shuchi, 陈心锐, 陈子恒, 冯浩, 胡桂鹏, 胡思超, 黄磊, 黄生帅,
黄勇, 黄子灵, 焦毅, 林鑫, 倪钰鑫, 彭康, 苏海波, 田凯, 李子垠, 杨星宇, 牛君豪, 张森镇, 张小雯, 张又文, 赵栢杨, 周胜凯,
朱国栋, 张银霞, 谭纬城，吴致远，杨星宇，刘群，曹宇麟，方泽华，金仁操，刘群，李永文，钱驾宏，吴昊天，杨璇，汪家傲

欢迎以任何形式对项目提供贡献！
