# MindSpore Transformers Release Notes

## MindSpore Transformers 1.8.0 Release Notes

The following outlines the key new features and bug fixes introduced in version 1.8.0 of the MindSpore Transformers suite, compared to version 1.7.0.

### New Features

- **Training Features:** Mcore models support fine-grained configuration parameters for [initialization standard deviation](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/configuration.html); learning rate strategy supports fine-grained configuration of [grouped learning rates](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/training_hyperparameters.html); new [Muon optimizer](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/training_hyperparameters.html) with QKClip configuration support, implementing [MuonClip optimizer](https://arxiv.org/pdf/2507.20534).
- **Mcore Model Architecture:** Support for different [position encoding](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/other_training_features.html?highlight=nope#%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81) strategies for different TransformerLayer configurations; support for configuring [SlidingWindowAttention](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/other_training_features.html?highlight=nope#slidingwindowattention).
- **Datasets:** Hugging Face datasets support streaming data loading, reducing dataset loading time for fine-tuning tasks.
- **Architecture Upgrade:** [Weight saving/loading](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/checkpoint_saving_and_loading.html) & [resume training](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/resume_training2.0.html) solution upgraded, implementing a new weight directory structure, simplified configuration, and Reshard loading mechanism, significantly improving usability and loading/recovery performance.

### Bugfix

During the current release cycle, we have implemented numerous bugfixes across models, functionalities, usability, and documentation. Key fixes include:

- [!7824](https://gitee.com/mindspore/mindformers/pulls/7874): Fixed issue where pad_token_id was not effective in Mcore networks.
- [!7818](https://gitee.com/mindspore/mindformers/pulls/7818): Fixed hostname retrieval failure issue in certain environments.
- [!7793](https://gitee.com/mindspore/mindformers/pulls/7793) [!7713](https://gitee.com/mindspore/mindformers/pulls/7713): Fixed Hugging Face dataset related issues.
- [!7630](https://gitee.com/mindspore/mindformers/pulls/7630): Fixed safetensors weight conversion and loading issue when changing parallel strategy.
- [!7743](https://gitee.com/mindspore/mindformers/pulls/7743): Fixed hidden_size assignment logic when shared experts are greater than 1.
- [!7790](https://gitee.com/mindspore/mindformers/pulls/7790): Fixed inference weight loading failure when q_lora_rank is None.
- [!7902](https://gitee.com/mindspore/mindformers/pulls/7902): Fixed error in DeepSeek-V3 inference model when weights are not loaded.

### Change Notes

This release introduces modifications to certain historical deprecated models/code/materials. Detailed changes and explanations are as follows:

| Change Content          | Change Description                                                   |
|:------------------------|:---------------------------------------------------------------------|
| Deprecated Model Sunset | The following models have been sunset: Llama3.1, Mixtral, Llm_boost. |

### Contributors

We extend our gratitude to the following team and their members for their outstanding contributions:

- **天翼云息壤智算团队:** [RFC](https://gitee.com/mindspore/mindformers/issues/IDCHDD) [!7757](https://gitee.com/mindspore/mindformers/pulls/7757) Support for MoE expert hot/cold expert migration, improving training performance during the initial phase of MoE model training when expert load is unbalanced.

We also thank the following developers who contributed during the release cycle:

[@ccsszz](https://gitee.com/ccsszz), [@chenrayray](https://gitee.com/chenrayray), [@hangangqiang](https://gitee.com/hangangqiang), [@highcloud3](https://gitee.com/highcloud3), [@hss-shuai](https://gitee.com/hss-shuai), [@huan-xiaoling](https://gitee.com/huan-xiaoling), [@husichao](https://gitee.com/husichao), [@jimmyisme](https://gitee.com/jimmyisme), [@JingweiHuang](https://gitee.com/JingweiHuang), [@lanshaozuishuai](https://gitee.com/lanshaozuishuai), [@limuan](https://gitee.com/limuan), [@Lin-Bert](https://gitee.com/Lin-Bert), [@liulili-huawei](https://gitee.com/liulili-huawei), [@liu-yanwei6](https://gitee.com/liu-yanwei6), [@lzy0920232](https://gitee.com/lzy0920232), [@minghu111](https://gitee.com/minghu111), [@niu-junhao01](https://gitee.com/niu-junhao01), [@pengjingyou](https://gitee.com/pengjingyou), [@qsc97](https://gitee.com/qsc97), [@renyujin](https://gitee.com/renyujin), [@senzhen-town](https://gitee.com/senzhen-town), [@smallsilly](https://gitee.com/smallsilly), [@Somnus2020](https://gitee.com/Somnus2020), [@song-jiaqi1999](https://gitee.com/song-jiaqi1999), [@suhaibo](https://gitee.com/suhaibo), [@Sunshine_Youngster](https://gitee.com/Sunshine_Youngster), [@wei_zhuoyi](https://gitee.com/wei_zhuoyi), [@xiaoqi-zhou](https://gitee.com/xiaoqi-zhou), [@yinanf](https://gitee.com/yinanf), [@yiyison](https://gitee.com/yiyison), [@yule100](https://gitee.com/yule100), [@zhangyihuiben](https://gitee.com/zhangyihuiben), [@zyw-hw](https://gitee.com/zyw-hw), [@zzzkeke](https://gitee.com/zzzkeke)

Contributions to the project in any form are most welcome!