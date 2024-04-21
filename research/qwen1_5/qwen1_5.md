# 通义千问

## 模型描述

通义千问是阿里云研发的通义千问大模型系列。Qwen1.5是Qwen2的beta版本, 基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。

```text
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```

## 仓库介绍

`Qwen1.5` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   qwen1_5
     └── qwen1_5_tokenizer.py          # tokenizer
   ```

2. 模型配置：

   ```text
   qwen1_5
     ├── predict_qwen1_5_14b.yaml          # 14B 在线推理启动配置
     └── predict_qwen1_5_72b.yaml          # 72B 在线推理启动配置
   ```

3. 环境准备和任务启动脚本：

   ```text
   qwen1_5
     ├── convert_weight.py             # 权重转换脚本
     └── run_qwen1_5.py                # Qwen高阶接口脚本
   ```

## 前期准备

### [mindformers安装](path/to/README.md#二mindformers安装)

### 环境要求

- 硬件：910B
- MindSpore：2.3
- MindFormers版本：dev
- Python：3.9

注：

环境搭建参考 [MindSpore官网](https://www.mindspore.cn/install/)，安装MindSpore及CANN社区版配套版本。

### 模型权重准备

#### torch权重转mindspore权重

从huggingface下载预训练权重后根据以下步骤进行权重转换，需要下载整个工程，huggingface权重的链接如下：

- [Qwen1.5-14B-Base](https://huggingface.co/Qwen/Qwen1.5-14B/tree/main)
- [Qwen1.5-72B-Base](https://huggingface.co/Qwen/Qwen1.5-72B/tree/main)

**注**: 请安装`convert_weight.py`依赖包。后续所用的vocab.json和merges.txt文件在此工程中获取。

```shell
pip install torch transformers transformers_stream_generator einops accelerate
# transformers版本不低于4.37.2
```

下载完成后，运行`convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python research/qwen1_5/convert_weight.py \
--torch_ckpt_dir <torch_ckpt_dir> \
--mindspore_ckpt_path <mindspore_ckpt_path>
# 参数说明：
# torch_ckpt_dir: 预训练权重文件所在的目录，此参数必须。
# mindspore_ckpt_path: 转换后的输出文件存放路径。可选，如果不给出，默认为`./transform.ckpt`
```

#### mindspore权重转torch权重

在生成mindspore权重之后如需使用torch运行，可根据如下命令转换：

```shell
python convert_reversed.py --mindspore_ckpt_path /path/your.ckpt --torch_ckpt_path /path/your.bin
# 参数说明：
# mindspore_ckpt_path: 待转换的mindspore权重，此参数必须。
# torch_ckpt_path: 转换后的输出文件存放路径，此参数必须。
```

### [模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## 推理

大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合大算子降低推理时延，有效提升网络吞吐量。

注意事项：

1. 当前支持模型已提供yaml文件，下文以Qwen1_5-72B为例，即使用`predict_qwen1_5_72b.yaml`配置文件进行介绍，请根据实际使用模型更改配置文件。

2. 运行下面的代码需要在`research/qwen1_5`目录下，或者先将`research/qwen1_5`目录所在路径加入到`PYTHONPATH`环境变量中。

### 基于高阶接口推理

- **多卡推理**

1. 主要参数配置参考：

   ```yaml
   load_checkpoint: '/path/model_dir'       # 使用切分完的权重
   auto_trans_ckpt: False                   # 打开自动权重转换
   use_past: True                           # 使用增量推理
   use_parallel: True                       # 使用并行模式

   model:
     model_config:
       use_past: True
       is_dynamic: True

   processor:
     tokenizer:
       vocab_file: "/{path}/vocab.json"     # vocab.json文件路径
       merges_file: "/{path}/merges.txt"    # merges.txt文件路径

   # parallel of device num = 2
   parallel_config:
     data_parallel: 1
     model_parallel: 4
     pipeline_stage: 1
     micro_batch_num: 1
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   ```

   *注*：可配置`model_config:param_init_type`为`float32`提高推理精度，但同时会影响在线推理性能。

2. 启动推理：

   ```shell
   cd mindformers/research/qwen1_5
   # 推理命令中参数会覆盖yaml文件中的相同参数
   bash ../../scripts/msrun_launcher.sh "python run_qwen1_5.py \
   --config predict_qwen1_5_72b.yaml \
   --load_checkpoint /path/model_dir \
   --run_mode predict \
   --use_parallel True \
   --predict_data 帮助我制定一份去上海的旅游攻略 \
   --auto_trans_ckpt False" 4

   # 帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息……
   ```
