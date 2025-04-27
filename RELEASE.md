## MindSpore Transformers Release Notes

## MindSpore Transformers 1.5.0 Release Notes

The following is the changelog for the MindSpore Transformers suite version 1.5.0, with the following key new features and bugfixes compared to version 1.3.2.

### New Features

* [Distributed Parallelism](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/distributed_parallel.html): Added Seq Pipe feature, Hybrid Sequence Parallelization feature.
* [Weights](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/safetensors.html): Added support for Safetensors format weights.
* [Training Monitor](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/monitor.html): Added support for TensorBoard real-time visualized monitoring of training metrics.
* [High Availability](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/high_availability.html): Added end-of-life CKPT function, UCE fault tolerance recovery function and process-level rescheduling recovery function.
* [Heterogeneous Storage](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/fine_grained_activations_swap.html): Added SWAP function for fine-grained activation values during training.

### New Models

The following new models are supported:

| Models                                                                                       | Specifications                                                                        |
|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| [DeepSeek-V3/R1](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/deepseek3)     | DeepSeek-V3-671B (pre-training, fine-tuning, inference), DeepSeek-R1-671B (inference) |
| [Llama3.2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama3_2.md) | Llama3.2-3B (inference), Llama3.2-Vision-11B (fine-tuning, inference)                 |
| [Qwen2.5](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/qwen2_5)              | Qwen2.5-0.5B/1.5B (inference) /7B/14B/32B/72B (fine-tuning, inference)                |
| [TeleChat2](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/telechat2)          | TeleChat2-7B/35B/115 (fine-tuning, inference)                                         |
| [YiZhao](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/yizhao)                | YiZhao-12B (pre-training, fine-tuning, inference)                                     |

### Bugfix

During the current release cycle, we have bugfixed many aspects of the model/functionality/usability/documentation. Here is a list of some of the key fixes:

* [!6013](https://gitee.com/mindspore/mindformers/pulls/6013): Fixed incompatibility between context parallelism (cp) and sequence parallelism (use_seq_parallel).
* [!6007](https://gitee.com/mindspore/mindformers/pulls/6007): Fixed that setting the maximum number of checkpoints to keep during training (keep_checkpoint_max) does not take effect on keeping checkpoints for pure model parameters.

### Change Description

In the current version, some historical deprecated models/codes/documentations have been changed. Details of the changes are as follows:

| Change Content                                                          | Change Description                                                                                                                                                                                                                                 |
|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Downgraded code, configuration files and materials of deprecated models | The models involved include Bloom, BaiChuan, BaiChuan2, CodeGeeX, CodeGeeX2, GLM, GLM2, VisualGLM, InternLM, PanguAlpha, SAM, SkyWork, WizardCoder, Qwen, Ziya, Llama                                                                              |
| Downgraded code for deprecated interfaces                               | The involved interfaces include CompareLoss, FusedCastAdamWeightDecay, MultiImgCapDataLoader, MultiImgCapDataset, ImageToTextRetrievalTrainer, auto_augment, group_ic_params, group_mim_parameters, TokenClassificationTrainer                     |
| Downgraded the old version of the official documentation                | Downgraded the old version of the documentation related files in the repository. Subsequent official documentation is available at [MindSpore Transformers Official Documentation](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/index.html) |

### Contributors

Thanks to the following people for their contributions:

chengxianbin, Chong Li, ehaleva, hangangqiang, huangshengshuai, huangzhuo, leida, lilei, limengyuan, liubuyu, lizhihao, moran, wangpingan, wangshaocong, wudawei, wutiancheng, wuweikang, yangminghai, yao_yf, zhanzhan, ZhouJingfeng, zhouyaqiang, 常少中, 陈心锐, 陈昱坤, 程泽睿志, 樊瑞, 范益, 封霆谚, 冯浩, 葛煜洪, 郭儒辰, 何泽泉, 胡安东, 胡思超, 胡志坤, 宦晓玲, 黄靖伟, 黄磊, 黄新元, 黄勇, 黄志超, 黄子灵, 季文尚, 金仁操, 孔紫怡, 蓝翔, 李嘉坤, 李俊标, 李子垠, 林盈来, 刘晨晖, 刘烙彬, 刘力力, 刘言伟, 马成贵, 倪钰鑫, 牛君豪, 彭竞由, 秦思莼, 任峪瑾, 赛尧, 苏海波, 孙宇轩, 谭纬城, 唐德志, 汪家傲, 王浩然, 王振邦, 魏琢艺, 吴昊天, 吴治锋, 吴致远, 肖尧, 尤日帆, 俞涵, 张丹阳, 张浩, 张敏利, 张森镇, 张奕晖, 张又文, 赵奕舜, 周声煦, 周小琪, 祝建伟, 邹文祥

Contributions to the project in any form are welcome!