# MindSpore Transformers Release Notes

## MindSpore Transformers 1.7.0 Release Notes

The following outlines the key new features and bug fixes introduced in version 1.7.0 of the MindSpore Transformers, compared to version 1.6.0.

### New Features

* Datasets: Hugging Face datasets now support column-specific reading and redundant data read I/O optimisation;
* Training: Support for PMA optimiser; optimiser state CPU offloading; group routing for MoE training; inter-machine communication merging for MoELayer;
* Inference: Support for A8W4/A8W8 quantisation inference; DeepSeek-V3/R1 models support MTP parallel inference; Mcore inference supports PP/EP parallelism.

### New Models

The following models are newly supported:

| Model                  | Specifications                                                                                       |
|:-----------------------|:-----------------------------------------------------------------------------------------------------|
| Qwen3 (Mcore)          | Qwen3-32B (Pre-training, Fine-tuning, Inference), Qwen3-0.6B/1.7B/4B/8B/14B (Fine-tuning, Inference) |
| Qwen3-MoE (Mcore)      | Qwen3-30B-A3B (Pre-training, Inference), Qwen3-235B-A22B (Inference)                                 |
| DeepSeek-V3/R1 (Mcore) | DeepSeek-V3-671B (Inference)                                                                         |
| TeleChat2 (Mcore)      | TeleChat2-7B/35B (Inference)                                                                         |

### Bugfix

During the current release cycle, we have implemented numerous bugfixes across models, functionalities, usability, and documentation. Key fixes include:

* [!7150](https://gitee.com/mindspore/mindformers/pulls/7150): Fixed incorrect generation count for Megatron dataset;
* [!7366](https://gitee.com/mindspore/mindformers/pulls/7366): Resolved weight validation error during scaling and resume training;
* [!7533](https://gitee.com/mindspore/mindformers/pulls/7533): Resolved loading anomalies when resuming training with specified Safetensors weights bearing identical suffixes;
* [!7397](https://gitee.com/mindspore/mindformers/pulls/7397): Resolved failure to run when aux_loss employed default values during training;
* [!7486](https://gitee.com/mindspore/mindformers/pulls/7486): Addressed accuracy issues when both CP and EP were enabled concurrently in Mcore architecture training scenarios;
* [!7507](https://gitee.com/mindspore/mindformers/pulls/7507): Resolved an issue where weights were saved abnormally during fault-tolerant recovery;
* [!6912](https://gitee.com/mindspore/mindformers/pulls/6912): Fixed a circular import issue during build_context initialisation;
* [!7513](https://gitee.com/mindspore/mindformers/pulls/7513): Resolved an issue where the number of TP exceeded the kv_head count during training weight loading in Mcore architecture inference scenarios;
* [!7247](https://gitee.com/mindspore/mindformers/pulls/7247): Fixed an issue where the Router module in Mcore architecture inference scenarios failed to activate fusion operators and routing algorithm selection based on configuration.

### Change Notes

This release introduces modifications to certain historical deprecated models/code/materials. Detailed changes and explanations are as follows:

| Change Content          | Change Description                                                                                                                  |
|:------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| Deprecated Model Sunset | The following models have commenced their sunset process: Llama3.1, Mixtral, Llm_boost. They will be discontinued in version 1.8.0. |

### Contributors

We extend our gratitude to the following individuals for their contributions:

dengyepeng, hangangqiang, huangshengshuai, huangzhuo, wangpingan, wangshaocong, zhanzhan, 常少中, 陈心锐, 陈昱坤, 封霆谚, 郭儒辰, 贺冬冬, 胡思超, 胡志坤, 宦晓玲, 黄靖伟, 霍新友, 金仁操, 孔紫怡, 蓝翔, 李惠兰, 李俊标, 李子垠, 刘烙彬, 刘通, 鲁力宁, 牛君豪, 彭竞由, 秦思莼, 任峪瑾, 赛尧, 苏海波, 万屹东, 魏琢艺, 肖尧, 许峰, 杨耀东, 尤日帆, 张森镇, 张奕晖, 张又文, 赵奕舜, 钟颢文, 周小琪, 朱晓晨

Contributions to the project in any form are most welcome!