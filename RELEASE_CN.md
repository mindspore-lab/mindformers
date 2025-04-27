# MindSpore Transformers Release Notes

## MindSpore Transformers 1.5.0 Release Notes

以下为MindSpore Transformers套件1.5.0版本的变更日志，相较于1.3.2版本有以下关键新特性和Bugfix。

### 新特性

* [分布式并行](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/function/distributed_parallel.html)：新增序列流水线并行（Seq Pipe）特性，新增混合序列并行特性。
* [权重](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/function/safetensors.html)：新增支持Safetensors格式权重。
* [训练监控](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/function/monitor.html)：新增支持TensorBoard训练指标实时可视化监控功能。
* [高可用](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/function/high_availability.html)：新增临终 CKPT 功能、UCE 故障容错恢复功能和进程级重调度恢复功能。
* [异构存储](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/function/fine_grained_activations_swap.html)：新增支持训练时细粒度的激活值SWAP功能。

### 新模型

以下为新支持模型：

| 模型                                                                                           | 规格                                               |
|----------------------------------------------------------------------------------------------|--------------------------------------------------|
| [DeepSeek-V3/R1](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/deepseek3)     | DeepSeek-V3-671B（预训练、微调、推理）、DeepSeek-R1-671B（推理） |
| [Llama3.2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama3_2.md) | Llama3.2-3B（推理）、Llama3.2-Vision-11B （微调、推理）      |
| [Qwen2.5](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/qwen2_5)              | Qwen2.5-0.5B/1.5B（推理）/7B/14B/32B/72B (微调、推理)     |
| [TeleChat2](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/telechat2)          | TeleChat2-7B/35B/115（微调、推理）                      |
| [YiZhao](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/yizhao)                | YiZhao-12B（预训练、微调、推理）                            |

### Bugfix

在当前版本发布周期内，我们进行了模型/功能/易用性/文档等诸多方面的 bugfix ，在此列举部分关键修复内容：

* [!6013](https://gitee.com/mindspore/mindformers/pulls/6013)：修复上下文并行（cp）与序列并行（use_seq_parallel）不兼容的问题。
* [!6007](https://gitee.com/mindspore/mindformers/pulls/6007)：修复训练时设置最多保留的checkpoint数量（keep_checkpoint_max）对保存纯模型参数的checkpoint不生效的问题。

### 变更说明

当前版本对部分历史的废弃模型/代码/资料进行了变更，详细的变更内容及说明如下：

| 变更内容               | 变更说明                                                                                                                                                                                         |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 下架了废弃模型的代码、配置文件和资料 | 涉及模型包含Bloom、BaiChuan、BaiChuan2、CodeGeeX、CodeGeeX2、GLM、GLM2、VisualGLM、InternLM、PanguAlpha、SAM、SkyWork、WizardCoder、Qwen、Ziya、Llama                                                             |
| 下架了废弃接口的代码         | 涉及接口包含CompareLoss、FusedCastAdamWeightDecay、MultiImgCapDataLoader、MultiImgCapDataset、ImageToTextRetrievalTrainer、auto_augment、group_ic_params、group_mim_parameters、TokenClassificationTrainer |
| 下架了老版本官方文档         | 下架了仓库内老版本文档相关文件。后续官方资料文档统一呈现在[MindSpore Transformers官方文档](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/index.html)                                                                 |

### 贡献者

感谢以下人员做出的贡献：

chengxianbin、Chong Li、ehaleva、hangangqiang、huangshengshuai、huangzhuo、leida、lilei、limengyuan、liubuyu、lizhihao、moran、wangpingan、wangshaocong、wudawei、wutiancheng、wuweikang、yangminghai、yao_yf、zhanzhan、ZhouJingfeng、zhouyaqiang、常少中、陈心锐、陈昱坤、程泽睿志、樊瑞、范益、封霆谚、冯浩、葛煜洪、郭儒辰、何泽泉、胡安东、胡思超、胡志坤、宦晓玲、黄靖伟、黄磊、黄新元、黄勇、黄志超、黄子灵、季文尚、金仁操、孔紫怡、蓝翔、李嘉坤、李俊标、李子垠、林盈来、刘晨晖、刘烙彬、刘力力、刘言伟、马成贵、倪钰鑫、牛君豪、彭竞由、秦思莼、任峪瑾、赛尧、苏海波、孙宇轩、谭纬城、唐德志、汪家傲、王浩然、王振邦、魏琢艺、吴昊天、吴治锋、吴致远、肖尧、尤日帆、俞涵、张丹阳、张浩、张敏利、张森镇、张奕晖、张又文、赵奕舜、周声煦、周小琪、祝建伟、邹文祥

欢迎以任何形式对项目提供贡献！