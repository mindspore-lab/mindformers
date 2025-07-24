# MindSpore Transformers Release Notes

## MindSpore Transformers 1.6.0 Release Notes

The following are the change logs for MindSpore Transformers suite version 1.6.0. Compared to version 1.5.0, there are the following key new features and bug fixes.

### New Features

* Model Architecture: The model architecture has been completely upgraded, encapsulating high-performance Transformer interfaces and providing a unified LLM model interface. Through the module spec mechanism, configurable model assembly is achieved. Mainstream LLM models can now be assembled and built using common modules, reducing redundant model code and enhancing the generalizability of functional features. All Transformer interfaces are aligned with Megatron-LM, and model training supports interface-level precision comparison.
* Community Collaboration: Supports use of Hugging Face model configurations, tokenizers, and model weights. Enables direct loading of Hugging Face model directories for inference.
* High availability: Training supports quick recovery without restarting. Without exiting the process, training can be resumed without re-executing time-consuming processes such as communication chain building and graph compilation. Training supports weight health detection and abnormal data skipping functions. It monitors specific indicators to determine weight health and supports skipping data that causes abnormal global norms. After multiple skips, training is automatically terminated, and training can be manually resumed from the previous healthy weight.
* Service-oriented: Qwen3-32B supports vLLM service-oriented deployment.
* Learning rate: Supports WSD (warmup stable decay) learning rate, which is a commonly used learning rate algorithm in current pre-training.
* Documentation: The official documentation structure has been optimized, with the outline structure adjusted for clearer logic and easier access to required content; provides a case study on model distillation using the DeepSeek-R1 model.

### New Models

The following are the newly supported models:

| Model                          | Specifications                               |
|:-------------------------------|:---------------------------------------------|
| DeepSeek-V3 (new architecture) | DeepSeek-V3-671B (pre-training, fine-tuning) |
| Qwen3 (new architecture)       | Qwen3-32B/235B (inference)                   |

### Bugfix

During the current version release cycle, we have made numerous bugfixes to the model, functionality, usability, documentation, and other aspects. Here are some of the key fixes:

* [!6575](https://gitee.com/mindspore/mindformers/pulls/6575): Fixed the issue of high host memory usage in the EOD compression scenario with CommonDataloader
* [!6568](https://gitee.com/mindspore/mindformers/pulls/6568): Fixed an error in the DropRateCallback during single-card training
* [!6209](https://gitee.com/mindspore/mindformers/pulls/6209): Fixed an issue where the init_method_std setting for shared experts in MoE scenarios was not taking effect
* [!6130](https://gitee.com/mindspore/mindformers/pulls/6130): Fixed an issue where the Megatron-LM dataset module was not packaged during build

### Change Notes

This version has made changes to some deprecated models/code/materials from previous versions. The detailed change contents and explanations are as follows:

| Change Content                                                           | Change Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|:-------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Removed code, configuration files, and documentation for obsolete models | Models involved include CodeLlama, Llama2, Llama3, Llama3.2, Llava, Llava-next, CogVLM2-Image, CogVLM2-Video, mLlama, Whisper, DeepSeek-V2, InternLM2, Qwen1.5, Qwen2, QwenVL, TeleChat, YiZhao, DeepSeek-Coder, DeepSeek-Coder-v1.5, GLM3, GLM32k, KnowLM, Yi, Bert, Clip, GPT2, ViT, MAE, Swin, T5, Blip2                                                                                                                                                                                                                                                                                                                                                                          |
| Removed code for deprecated interfaces                                   | Interfaces involved include MaskedLanguageModelingTrainer, QuestionAnsweringTrainer, TextClassificationTrainer, FillMaskPipeline, ZeroShotImageClassificationTrainer, ConstrastiveLanguageImagePretrainTrainer, MaskedImageModelingTrainer, MaskedImageModelingPipeline, MIMDataset, SimMask, MaeMask, ImageClassificationTrainer, ImageClassificationPipeline, ImageCLSDataset, TranslationTrainer, TranslationPipeline, TranslationDataset, WMT16DataLoader, ImageToTextGenerationTrainer, ImageToTextPipeline, Mixup, RandomErasing, text_transform, SoftTargetCrossEntropy, MSELoss, L1Loss, SQuADMetric, FusedAdamWeightDecay, FP32StateAdamWeightDecay, hccl_tools, merge_hccl |
| Deprecated features removed                                              | Features affected include Chat Web                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Deprecated old version official documentation                            | Deprecated old version documentation files in the repository have been removed. Going forward, official documentation will be uniformly presented in the MindSpore Transformers official documentation                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

### Contributors

We would like to thank the following individuals for their contributions:

chengxianbin, dengyepeng, ehaleva, heqinglin, huangshengshuai, huangzhuo, leida, lilei, liubuyu, wangpingan, wangshaocong, wuweikang, xiaruijie, yangminghai, zhangxuetong, zhanzhan, 常少中, 陈心锐, 陈昱坤, 樊瑞, 封霆谚, 葛煜洪, 郭儒辰, 郭志斌, 胡安东, 胡铭, 胡思超, 黄靖伟, 黄磊, 黄勇, 吉荣庭, 纪泽伟, 金仁操, 孔紫怡, 蓝翔, 雷赐晨, 李俊标, 李子垠, 林盈来, 刘晨晖, 刘烙彬, 刘力力, 刘言伟, 牛君豪, 彭竞由, 秦思莼, 任峪瑾, 赛尧, 苏海波, 孙宇轩, 谭纬城, 汪家傲, 王泓皓, 王振邦, 魏琢艺, 吴昊天, 吴治锋, 吴致远, 肖尧, 杨耀东, 易阳, 尤日帆, 俞涵, 张森镇, 张奕晖, 张又文, 赵奕舜, 周小琪, 祝建伟

We welcome contributions to the project in any form!