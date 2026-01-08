# MindSpore Transformers Release Notes

## MindSpore Transformers 1.8.0 Release Notes

以下为MindSpore Transformers套件1.8.0版本的变更日志，相较于1.7.0版本有以下关键新特性和bugfix。

### 新特性

- **训练功能：** Mcore模型支持细粒度配置参数[初始化标准差](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/configuration.html)；学习率策略支持细粒度配置[分组学习率](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/training_hyperparameters.html)；新增[Muon优化器](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/training_hyperparameters.html)，支持配置QKClip，实现[MuonClip优化器](https://arxiv.org/pdf/2507.20534)。
- **Mcore模型结构：** 支持不同TransformerLayer配置不同[位置编码](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/other_training_features.html?highlight=nope#%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81)策略；支持配置[SlidingWindowAttention](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/other_training_features.html?highlight=nope#slidingwindowattention)。
- **数据集：** Hugging Face数据集支持流式加载数据，降低微调任务的数据集加载时长。
- **架构升级：**[权重保存加载](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/checkpoint_saving_and_loading.html) & [断点续训](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/resume_training2.0.html) 方案升级，实现全新权重目录结构、配置简化及Reshard加载机制，显著提升易用性及加载/恢复性能。

### Bugfix

在当前版本发布周期内，我们进行了模型/功能/易用性/文档等诸多方面的bugfix，在此列举部分关键修复内容：

- [!7824](https://gitee.com/mindspore/mindformers/pulls/7874)：修复Mcore网络中pad_token_id不生效问题。
- [!7818](https://gitee.com/mindspore/mindformers/pulls/7818)：修复部分环境下hostname获取失败问题。
- [!7793](https://gitee.com/mindspore/mindformers/pulls/7793) [!7713](https://gitee.com/mindspore/mindformers/pulls/7713)：修复Hugging Face数据集相关问题。
- [!7630](https://gitee.com/mindspore/mindformers/pulls/7630)：修复变换并行策略时safetensors权重转换加载问题。
- [!7743](https://gitee.com/mindspore/mindformers/pulls/7743)：修复共享专家大于1场景下hidden_size赋值逻辑。
- [!7790](https://gitee.com/mindspore/mindformers/pulls/7790)：修复q_lora_rank为None时，推理权重加载失败的问题。
- [!7902](https://gitee.com/mindspore/mindformers/pulls/7902)：修复DeepSeek-V3推理模型不加载权重场景的报错。

### 变更说明

当前版本对部分历史的废弃模型/代码/资料进行了变更，详细的变更内容及说明如下：

| 变更内容   | 变更说明                                           |
|:-------|:-----------------------------------------------|
| 废弃模型下架 | 以下模型已下架：Llama3.1、Mixtral、Llm_boost。            |

### 贡献者

感谢以下团队人员做出的突出贡献：

- **天翼云息壤智算团队：** [RFC](https://gitee.com/mindspore/mindformers/issues/IDCHDD) [!7757](https://gitee.com/mindspore/mindformers/pulls/7757) 支持MoE专家冷热专家迁移，提升MoE模型训练初期专家负载不均衡时的训练性能。

感谢以下所有在版本周期内参与贡献的开发者：

[@ccsszz](https://gitee.com/ccsszz)、[@chenrayray](https://gitee.com/chenrayray)、[@hangangqiang](https://gitee.com/hangangqiang)、[@highcloud3](https://gitee.com/highcloud3)、[@hss-shuai](https://gitee.com/hss-shuai)、[@huan-xiaoling](https://gitee.com/huan-xiaoling)、[@husichao](https://gitee.com/husichao)、[@jimmyisme](https://gitee.com/jimmyisme)、[@JingweiHuang](https://gitee.com/JingweiHuang)、[@lanshaozuishuai](https://gitee.com/lanshaozuishuai)、[@limuan](https://gitee.com/limuan)、[@Lin-Bert](https://gitee.com/Lin-Bert)、[@liulili-huawei](https://gitee.com/liulili-huawei)、[@liu-yanwei6](https://gitee.com/liu-yanwei6)、[@lzy0920232](https://gitee.com/lzy0920232)、[@minghu111](https://gitee.com/minghu111)、[@niu-junhao01](https://gitee.com/niu-junhao01)、[@pengjingyou](https://gitee.com/pengjingyou)、[@qsc97](https://gitee.com/qsc97)、[@renyujin](https://gitee.com/renyujin)、[@senzhen-town](https://gitee.com/senzhen-town)、[@smallsilly](https://gitee.com/smallsilly)、[@Somnus2020](https://gitee.com/Somnus2020)、[@song-jiaqi1999](https://gitee.com/song-jiaqi1999)、[@suhaibo](https://gitee.com/suhaibo)、[@Sunshine_Youngster](https://gitee.com/Sunshine_Youngster)、[@wei_zhuoyi](https://gitee.com/wei_zhuoyi)、[@xiaoqi-zhou](https://gitee.com/xiaoqi-zhou)、[@yinanf](https://gitee.com/yinanf)、[@yiyison](https://gitee.com/yiyison)、[@yule100](https://gitee.com/yule100)、[@zhangyihuiben](https://gitee.com/zhangyihuiben)、[@zyw-hw](https://gitee.com/zyw-hw)、[@zzzkeke](https://gitee.com/zzzkeke)

欢迎以任何形式对项目提供贡献！