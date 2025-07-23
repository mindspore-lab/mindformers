# MindSpore Transformers Release Notes

## MindSpore Transformers 1.6.0 Release Notes

以下为MindSpore Transformers套件1.6.0版本的变更日志，相较于1.5.0版本有以下关键新特性和Bugfix。

### 新特性

* 模型架构：模型架构全新升级，封装高性能Transformer接口，提供LLM统一模型接口，通过module spec机制实现配置化模型组装。实现主流LLM模型可由公共模块组装搭建，减少冗余模型代码，增加功能特性的泛化性。其中所有Transformer接口与Megatron-LM进行了对齐，模型训练支持接口级精度比对。
* 社区协同：支持复用Hugging Face模型配置、分词器、模型权重。实现直接加载Hugging Face模型目录即可进行推理。
* 高可用：训练支持不重启快速恢复，在进程不退出的情况下，无需重新执行通信建链、图编译等耗时流程即可恢复训练；训练支持权重健康检测和异常数据跳过功能，通过监控特定指标判断权重健康性，支持跳过导致异常global norm的数据，多步跳过后自动终止训练，可手动从上一个健康的权重恢复训练。
* 服务化：Qwen3-32B支持vLLM服务化部署。
* 学习率：支持WSD（warmup stable decay）学习率，其为当前预训练常用的学习率算法。
* 文档资料：官方文档结构优化，调整了大纲结构使逻辑更清晰，方便查找所需内容；提供DeepSeek-R1模型蒸馏的案例。

### 新模型

以下为新支持模型：

| 模型               | 规格                       |
|:-----------------|:-------------------------|
| DeepSeek-V3（新架构） | DeepSeek-V3-671B（预训练、微调） |
| Qwen3（新架构）       | Qwen3-32B/235B（推理）       |

### Bugfix

在当前版本发布周期内，我们进行了模型/功能/易用性/文档等诸多方面的 bugfix ，在此列举部分关键修复内容：

* [!6575](https://gitee.com/mindspore/mindformers/pulls/6575)：修复了CommonDataloader在EOD压缩场景下Host显存占用大的问题
* [!6568](https://gitee.com/mindspore/mindformers/pulls/6568)：修复了单卡训练DropRateCallback报错
* [!6209](https://gitee.com/mindspore/mindformers/pulls/6209)：修复了MoE场景共享专家设置init_method_std不生效的问题
* [!6130](https://gitee.com/mindspore/mindformers/pulls/6130)：修复了构建时未将Megatron-LM数据集模块打包的问题

### 变更说明

当前版本对部分历史的废弃模型/代码/资料进行了变更，详细的变更内容及说明如下：

| 变更内容               | 变更说明                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|:-------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 下架了废弃模型的代码、配置文件和资料 | 涉及模型包括CodeLlama、Llama2、Llama3、Llama3.2、Llava、Llava-next、CogVLM2-Image、CogVLM2-Video、mLlama、Whisper、DeepSeek-V2、InternLM2、Qwen1.5、Qwen2、QwenVL、TeleChat、YiZhao、DeepSeek-Coder、DeepSeek-Coder-v1.5、GLM3、GLM32k、KnowLM、Yi、Bert、Clip、GPT2、ViT、MAE、Swin、T5、Blip2                                                                                                                                                                                                                                                                                                                                                                      |
| 下架了废弃接口的代码         | 涉及接口包括MaskedLanguageModelingTrainer、QuestionAnsweringTrainer、TextClassificationTrainer、FillMaskPipeline、ZeroShotImageClassificationTrainer、ConstrastiveLanguageImagePretrainTrainer、MaskedImageModelingTrainer、MaskedImageModelingPipeline、MIMDataset、SimMask、MaeMask、ImageClassificationTrainer、ImageClassificationPipeline、ImageCLSDataset、TranslationTrainer、TranslationPipeline、TranslationDataset、WMT16DataLoader、ImageToTextGenerationTrainer、ImageToTextPipeline、Mixup、RandomErasing、text_transform、SoftTargetCrossEntropy、MSELoss、L1Loss、SQuADMetric、FusedAdamWeightDecay、FP32StateAdamWeightDecay、hccl_tools、merge_hccl |
| 下架了废弃功能            | 涉及功能包括Chat Web                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 下架了老版本官方文档         | 下架了仓库内老版本文档相关文件。后续官方资料文档统一呈现在MindSpore Transformers官方文档                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

### 贡献者

感谢以下人员做出的贡献：

chengxianbin、dengyepeng、ehaleva、heqinglin、huangshengshuai、huangzhuo、leida、lilei、liubuyu、wangpingan、wangshaocong、wuweikang、xiaruijie、yangminghai、zhangxuetong、zhanzhan、常少中、陈心锐、陈昱坤、樊瑞、封霆谚、葛煜洪、郭儒辰、郭志斌、胡安东、胡铭、胡思超、黄靖伟、黄磊、黄勇、吉荣庭、纪泽伟、金仁操、孔紫怡、蓝翔、雷赐晨、李俊标、李子垠、林盈来、刘晨晖、刘烙彬、刘力力、刘言伟、牛君豪、彭竞由、秦思莼、任峪瑾、赛尧、苏海波、孙宇轩、谭纬城、汪家傲、王泓皓、王振邦、魏琢艺、吴昊天、吴治锋、吴致远、肖尧、杨耀东、易阳、尤日帆、俞涵、张森镇、张奕晖、张又文、赵奕舜、周小琪、祝建伟

欢迎以任何形式对项目提供贡献！