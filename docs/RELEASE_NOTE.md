# MindSpore Transformers 0.8.0 RELEASE NOTE

本文为MindSpore Transformers (以下称为MindFormers) 套件 0.8.0 版本的变更日志

## 新特性

- [权重自动切分合并特性](./feature_cards/Transform_Ckpt.md)：
    目前分布式训练/推理，当预训练权重与分布式策略不匹配时，需要将预训练权重转换为对应分布式策略的权重；本特性提供了相应权重转换的指导，并提供了自动完成权重转换的脚本，节省了用户该部分的工作量。
- [Text Generator功能优化重构](./feature_cards/Text_Generator.md):
    对text generator文本生成模块进行了功能性优化重构；标准化文本生成的后处理流程，支持top_k，top_p，temperature，repetition_penalty，max_new_tokens等后处理入参；支持增量推理，流式推理，batch推理等功能特性。
- [Chat Web Demo](./feature_cards/Chat_Web.md)：
    Chat Web提供了一个网页界面，让用户可以通过类似线上聊天的方式使用MindFormers大语言模型（LLM）推理能力，能够更加直观地展示模型推理效果。
- [MindSpore Lite离线推理能力](./feature_cards/Inference.md)：
    MindFormers定位打造训练->微调->部署的端到端大模型工具套件，为了更好性能地部署已经微调训练好的大模型；我们利用MindSpore打造的推理引擎[MindSpore Lite](https://www.mindspore.cn/lite)，为用户提供了开箱即用的推理部署方案，为用户提供端到端的大模型解决方案，帮助用户使能大模型业务。
- 保存不带优化器的权重：
    在保存权重时，额外保存一份不带优化器参数的权重，以避免因原权重体积过大，需要进行权重裁剪才能由于后续评估/推理；该功能通过配置文件中的 `callbacks` `CheckpointMointor` 下的 `save_network_params` 配置项控制，默认值为True表示打开，详情可参考[configs/README.md](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/README.md);
    后续计划将保存权重的形式进行可配置化，包括 带优化器参数/仅模型权重/仅可训练部分权重  等形式。
- 分布式推理：
    对于单卡显存无法完全加载的大模型，支持模型并行维度切分以在多卡上完成分布式推理；目前支持数据并行(data parallel，简称dp)/模型并行(model parallel，简称mp)的并行推理，建议仅使用mp切分以进行大模型的分布式推理，dp并行可使用多batch推理替换；正在进行pipeline推理相关的适配工作。
- 序列并行：
    支持序列并行的切分策略，可在模型seq_length维度进行切分，以支持长文本序列的训练；在配置项中使用 `use_seq_parallel` 以控制是否使用序列并行，需模型适配。
- 选择重计算：
    支持选择性开启[重计算](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.recompute)策略，只针对attention层的算子进行重计算，以减小开启选择重计算时带来的性能损耗；在配置项中使用 `recompute_config` 下的 `select_recompute` 配置项以控制是否开启选择重计算，需模型适配。
- 模型编译加速：
    集成了MindSpore的[lazy_inline](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/mindspore/mindspore.lazy_inline.html)子图复用特性，对存在可复用子图的模型进行编译加速。在MindFormers中，可通过 `export ENABLE_CELL_REUSE=1` 配置环境变量以开启编译加速功能，需模型适配。

## 新模型

| 模型                                      | 规格                                                                                                                          |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [GLM2](./model_cards/glm2.md)             | glm2_6b<br/>glm_b_lora                                                                                                        |
| [LLaMA2](./model_cards/llama2.md)         | llama2_7b<br/>llama2_13b                                                                                                      |
| [BLIP2](./model_cards/blip2.md)           | blip2_stage1_vit_g<br/>blip2_stage1_classification<br/>itt_blip2_stage2_vit_g_baichuan_7b<br/>itt_blip2_stage2_vit_g_llama_7b |
| [PanguAlpha](./model_cards/pangualpha.md) | pangualpha_2_6b<br/>pangualpha_13b                                                                                            |

以下为 research 模型：

| 模型                                             | 规格                                  |
| ------------------------------------------------ | ------------------------------------- |
| [Baichuan](../research/baichuan/baichuan.md)     | baichuan_7b<br/>baichuan_13b          |
| [Baichuan2](../research/baichuan2/baichuan2.md)  | baichuan2_7b<br/>baichuan2_13b        |
| [InternLM](../research/internlm/internlm.md)     | internlm_7b<br/>internlm_7b_lora      |
| [SAM](../research/sam/Segment_Anything_Model.md) | sam_vit_b<br/>sam_vit_l<br/>sam_vit_h |
| [Ziya](../research/ziya/ziya.md)                 | ziya_13b                              |

## 接口变更

我们对低参微调的适配接口进行了重构变更：

原版本每个模型需要在代码层实现一个 `XXXModelWithLora` 的模型结构以实现低参微调算法；现在我们将低参微调模型的获取接口统一为了 `get_pet_model(model, pet_config)` 接口，传入基础模型实例和微调配置即可获取低参微调模型的实例。详见[低参微调文档](./feature_cards/Pet_Tuners.md)以及各模型文档中低参微调相关部分。

此外，重构后的低参微调接口需 `mindpet==1.0.2` 的python依赖项，已在requirements.txt中更新，请用户注意版本配套更新。

## Bugfix

在当前版本发布周期内，我们进行了模型/功能/易用性/文档等诸多方面的Bugfix，在此仅列举部分修复内容：

- [#I82LJT] 修复了MindSpore 2.2版本上GELU激活函数的shard方法失效的问题。
- [#I8GHAW] 修复了模型加载权重不匹配时仍继续训练的问题。
- [#I86IXA] 修复了日志的性能数据在数据并行模式下计算错误的问题。
- [#I8BI31] 修复了pipeline流程在实例化模型时没有正确调用get_pet_model接口的问题。
- [#I7TKI3] 修复了使用ModelArts集群训练时，回传进程过多导致OBS访问数超限的问题。

欢迎对本项目提出意见与建议，以帮助项目持续改进。

## 其他

**文档**：重新规范化了当前存量的大多数文档与API说明，并上线了文档官网：[MindFormers 文档](https://mindformers.readthedocs.io/zh_CN/r0.8/)

**标准镜像**：随版本发布，同时发布标准docker镜像与相应的dockerfile文件，用户可使用标准镜像以获取MindFormers与相应版本配套的MindSpore环境依赖，以减少环境搭建的工作量以及规避可能遇到的各种环境问题。

## 贡献者

感谢以下人员做出的贡献：

Chenhua Geng, chenweifeng, Lin, renyujin, wanghua, wangxudong, Zhenhao Li, ZhidanLie, zhouyaqiang, 陈心锐, 陈子恒, 冯浩，耿辰华, 胡思超, 黄磊, 黄生帅, 黄欣靓, 黄勇, 黄子灵, 刘烙彬, 林鑫, 倪钰鑫, 钱驾宏, 苏海波, 田凯，余金, 张浩勇, 张鹤译, 张森镇, 张又文, 周胜凯, 朱国栋

欢迎以任何形式对项目提供贡献！
