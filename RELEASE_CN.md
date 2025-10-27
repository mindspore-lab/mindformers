# MindSpore Transformers Release Notes

## MindSpore Transformers 1.7.0 Release Notes

以下为MindSpore Transformers套件1.7.0版本的变更日志，相较于1.6.0版本有以下关键新特性和bugfix。

### 新特性

* 数据集：Hugging Face数据集支持指定数据列读取，支持数据读取IO去冗余；
* 训练功能：支持PMA优化器；优化器状态支持CPU offloading；MoE训练支持分组路由；MoELayer支持机间通信合并；
* 推理功能：支持A8W4/A8W8量化推理；DeepSeek-V3/R1模型支持MTP并行推理；Mcore推理支持PP/EP并行；

### 新模型

以下为新支持模型：

| 模型                    | 规格                                                    |
|:----------------------|:------------------------------------------------------|
| Qwen3（Mcore）          | Qwen3-32B（预训练、微调、推理）、Qwen3-0.6B/1.7B/4B/8B/14B（微调、推理） |
| Qwen3-MoE（Mcore）      | Qwen3-30B-A3B（预训练、推理）、Qwen3-235B-A22B（推理）             |
| DeepSeek-V3/R1（Mcore） | DeepSeek-V3-671B（推理）                                  |
| TeleChat2（Mcore）      | TeleChat2-7B/35B（推理）                                  |

### Bugfix

在当前版本发布周期内，我们进行了模型/功能/易用性/文档等诸多方面的bugfix，在此列举部分关键修复内容：

* [!7150](https://gitee.com/mindspore/mindformers/pulls/7150): 修复Megatron数据集生成数量错误问题；
* [!7366](https://gitee.com/mindspore/mindformers/pulls/7366): 修复扩容续训时权重校验错误的问题；
* [!7533](https://gitee.com/mindspore/mindformers/pulls/7533): 修复指定Safetensors权重续训时，遇到相同后缀Safetensors加载异常的问题；
* [!7397](https://gitee.com/mindspore/mindformers/pulls/7397): 修复aux_loss使用默认值进行训练时，无法运行的问题；
* [!7486](https://gitee.com/mindspore/mindformers/pulls/7486): 修复Mcore架构训练场景CP与EP同时开启时的精度问题；
* [!7507](https://gitee.com/mindspore/mindformers/pulls/7507): 修复故障快恢中保存权重异常的问题；
* [!6912](https://gitee.com/mindspore/mindformers/pulls/6912)：修复build_context初始化时的循环导入问题；
* [!7513](https://gitee.com/mindspore/mindformers/pulls/7513)：修复Mcore架构推理场景加载训练权重时TP数大于kv_head数场景的问题；
* [!7247](https://gitee.com/mindspore/mindformers/pulls/7247)：修复Mcore架构推理场景Router模块无法根据配置选择融合算子和路由算法激活函数的问题。

### 变更说明

当前版本对部分历史的废弃模型/代码/资料进行了变更，详细的变更内容及说明如下：

| 变更内容   | 变更说明                                               |
|:-------|:---------------------------------------------------|
| 废弃模型日落 | 以下模型开始日落流程：Llama3.1、Mixtral、Llm_boost。将在1.8.0版本下架。 |

### 贡献者

感谢以下人员做出的贡献：

dengyepeng、hangangqiang、huangshengshuai、huangzhuo、wangpingan、wangshaocong、zhanzhan、常少中、陈心锐、陈昱坤、封霆谚、郭儒辰、贺冬冬、胡思超、胡志坤、宦晓玲、黄靖伟、霍新友、金仁操、孔紫怡、蓝翔、李惠兰、李俊标、李子垠、刘烙彬、刘通、鲁力宁、牛君豪、彭竞由、秦思莼、任峪瑾、赛尧、苏海波、万屹东、魏琢艺、肖尧、许峰、杨耀东、尤日帆、张森镇、张奕晖、张又文、赵奕舜、钟颢文、周小琪、朱晓晨

欢迎以任何形式对项目提供贡献！