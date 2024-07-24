# MindSpore Transformers 1.2.0 RELEASE NOTE

以下为MindSpore Transformers (以下称为MindFormers) 套件 1.2.0 版本的变更日志，相较于1.1.0版本有以下关键新特性和bug
fix。

## 新特性

- **新增模型支持带框架推理**：新增支持模型包含Qwen1.5_7b、Qwen1.5_14b、Qwen1.5_72b、Llama3_70b、Yi_34b等。
- **新增模型支持bfloat16训练**：新增支持模型包含Qwen1.5_7b、Qwen1.5_14b、Qwen1.5_72b、Llama3_70b、Yi_34b等。
- [AdamW优化器](https://gitee.com/mindspore/mindformers/pulls/3310)：新增AdamW优化器，对齐Megatron AdamW。
- **支持MindIE进行服务化部署**：[MindIE](https://www.hiascend.com/software/mindie)，全称Mind Inference
  Engine，是华为昇腾针对AI全场景业务的推理加速套件。MindFormers新增对MindIE的对接，承载在模型应用层MindIE-LLM，通过MindIE-Service对MindFormers中LLM模型进行部署。
- [长序列训练](https://gitee.com/mindspore/mindformers/tree/r1.2.0/docs/feature_cards/Long_Sequence_Training.md)：新增支持长序列训练特性，通过在配置yaml文件中设置`parallel_config.context_parallel`开启序列并行，当前支持32k至256k。
- [断点续训权重加载2.0](https://gitee.com/mindspore/mindformers/tree/r1.2.0/docs/feature_cards/Resume_Training.md)：断点续训场景下，新增指定续训权重功能，新增故障恢复下进行权重完整性校验并自动加载最新完整权重。
- [权重自动转换2.0](https://gitee.com/mindspore/mindformers/tree/r1.2.0/docs/feature_cards/Transform_Ckpt.md)：自动权重转换新增多进程转换。

## 新模型

以下为新支持模型：

| 模型                                                                                            | 规格                                                            |
|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [Mixtral](https://gitee.com/mindspore/mindformers/tree/r1.2.0/research/mixtral/mixtral.md)    | Mixtral_8x7b（32k预训练、推理）                                       |
| [Qwen1.5](https://gitee.com/mindspore/mindformers/tree/r1.2.0/research/qwen1_5/qwen1_5.md)    | Qwen1.5_7b（预训练、微调、推理）、Qwen1.5_14b（预训练、微调、推理）、Qwen1.5_72b（预训练） |
| [Llama3](https://gitee.com/mindspore/mindformers/tree/r1.2.0/research/llama3/llama3.md)       | Llama3_70b（预训练、微调）                                            |
| [Deepseek](https://gitee.com/mindspore/mindformers/tree/r1.2.0/research/deepseek/deepseek.md) | Deepseek_33b（微调）                                              |
| [Yi](https://gitee.com/mindspore/mindformers/tree/r1.2.0/research/yi/yi.md)                   | Yi_6b（微调）、Yi_34b（微调）                                          |
| [QwenVL](https://gitee.com/mindspore/mindformers/tree/r1.2.0/research/qwenvl/qwenvl.md)       | QwenVL_9.6b（微调、推理）                                            |

## Bugfix

在当前版本发布周期内，我们进行了模型/功能/易用性/文档等诸多方面的Bugfix，在此仅列举部分修复内容：

- [!3345](https://gitee.com/mindspore/mindformers/pulls/3345)：修复`Linear`在`transpose_b=False`时传入正确`weight`仍报错问题。
- [!3277](https://gitee.com/mindspore/mindformers/pulls/3277)：修复使用梯度累积时，`warpper`传入了错误的`micro_batch_num`问题。

## 贡献者

感谢以下人员做出的贡献：

Chenhua Geng，heqinglin，koukairui，renyujin，shuchi，陈心锐，陈子恒，冯浩，胡思超，黄磊，黄生帅，黄勇，黄子灵，倪钰鑫，苏海波，李子垠，杨星宇，牛君豪，张森镇，张又文，谭纬城，吴致远，杨星宇，刘群，曹宇麟，方泽华，金仁操，刘群，李永文，钱驾宏，吴昊天，杨璇，汪家傲，范益，陈昱坤，李洋

欢迎以任何形式对项目提供贡献！
