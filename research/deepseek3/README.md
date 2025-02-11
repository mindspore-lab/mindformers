# DeepSeek-V3

## 模型描述

DeepSeek-V3是由DeepSeek（深度求索）推出的一个强大的专家混合（MoE）语言模型，它拥有671B总参数，其中激活参数量为37B。为了实现高效推理和低成本训练，DeepSeek-V3采用了多头潜注意力（MLA）和DeepSeekMoE架构，这在DeepSeek-V2中得到了充分验证。此外，DeepSeek-V3 还率先采用了无辅助损失的负载均衡策略，并设定了多token预测训练目标，以提高性能。DeepSeek-V3在14.8万亿个多种类的高质量token上进行预训练，接着通过监督微调和强化学习充分优化其能力。综合评估显示，在发布时DeepSeek-V3的性能优于其他开源模型，并可与领先的闭源模型相媲美。尽管性能卓越，DeepSeek-V3 的全部训练成本非常低，且其训练过程也非常稳定。

```text
@misc{deepseekai2024deepseekv3technicalreport,
      title={DeepSeek-V3 Technical Report},
      author={DeepSeek-AI and Aixin Liu and Bei Feng and Bing Xue and Bingxuan Wang and Bochao Wu and Chengda Lu and Chenggang Zhao and Chengqi Deng and Chenyu Zhang and Chong Ruan and Damai Dai and Daya Guo and Dejian Yang and Deli Chen and Dongjie Ji and Erhang Li and Fangyun Lin and Fucong Dai and Fuli Luo and Guangbo Hao and Guanting Chen and Guowei Li and H. Zhang and Han Bao and Hanwei Xu and Haocheng Wang and Haowei Zhang and Honghui Ding and Huajian Xin and Huazuo Gao and Hui Li and Hui Qu and J. L. Cai and Jian Liang and Jianzhong Guo and Jiaqi Ni and Jiashi Li and Jiawei Wang and Jin Chen and Jingchang Chen and Jingyang Yuan and Junjie Qiu and Junlong Li and Junxiao Song and Kai Dong and Kai Hu and Kaige Gao and Kang Guan and Kexin Huang and Kuai Yu and Lean Wang and Lecong Zhang and Lei Xu and Leyi Xia and Liang Zhao and Litong Wang and Liyue Zhang and Meng Li and Miaojun Wang and Mingchuan Zhang and Minghua Zhang and Minghui Tang and Mingming Li and Ning Tian and Panpan Huang and Peiyi Wang and Peng Zhang and Qiancheng Wang and Qihao Zhu and Qinyu Chen and Qiushi Du and R. J. Chen and R. L. Jin and Ruiqi Ge and Ruisong Zhang and Ruizhe Pan and Runji Wang and Runxin Xu and Ruoyu Zhang and Ruyi Chen and S. S. Li and Shanghao Lu and Shangyan Zhou and Shanhuang Chen and Shaoqing Wu and Shengfeng Ye and Shengfeng Ye and Shirong Ma and Shiyu Wang and Shuang Zhou and Shuiping Yu and Shunfeng Zhou and Shuting Pan and T. Wang and Tao Yun and Tian Pei and Tianyu Sun and W. L. Xiao and Wangding Zeng and Wanjia Zhao and Wei An and Wen Liu and Wenfeng Liang and Wenjun Gao and Wenqin Yu and Wentao Zhang and X. Q. Li and Xiangyue Jin and Xianzu Wang and Xiao Bi and Xiaodong Liu and Xiaohan Wang and Xiaojin Shen and Xiaokang Chen and Xiaokang Zhang and Xiaosha Chen and Xiaotao Nie and Xiaowen Sun and Xiaoxiang Wang and Xin Cheng and Xin Liu and Xin Xie and Xingchao Liu and Xingkai Yu and Xinnan Song and Xinxia Shan and Xinyi Zhou and Xinyu Yang and Xinyuan Li and Xuecheng Su and Xuheng Lin and Y. K. Li and Y. Q. Wang and Y. X. Wei and Y. X. Zhu and Yang Zhang and Yanhong Xu and Yanhong Xu and Yanping Huang and Yao Li and Yao Zhao and Yaofeng Sun and Yaohui Li and Yaohui Wang and Yi Yu and Yi Zheng and Yichao Zhang and Yifan Shi and Yiliang Xiong and Ying He and Ying Tang and Yishi Piao and Yisong Wang and Yixuan Tan and Yiyang Ma and Yiyuan Liu and Yongqiang Guo and Yu Wu and Yuan Ou and Yuchen Zhu and Yuduan Wang and Yue Gong and Yuheng Zou and Yujia He and Yukun Zha and Yunfan Xiong and Yunxian Ma and Yuting Yan and Yuxiang Luo and Yuxiang You and Yuxuan Liu and Yuyang Zhou and Z. F. Wu and Z. Z. Ren and Zehui Ren and Zhangli Sha and Zhe Fu and Zhean Xu and Zhen Huang and Zhen Zhang and Zhenda Xie and Zhengyan Zhang and Zhewen Hao and Zhibin Gou and Zhicheng Ma and Zhigang Yan and Zhihong Shao and Zhipeng Xu and Zhiyu Wu and Zhongyu Zhang and Zhuoshu Li and Zihui Gu and Zijia Zhu and Zijun Liu and Zilin Li and Ziwei Xie and Ziyang Song and Ziyi Gao and Zizheng Pan},
      year={2024},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.19437},
}
```

## 模型文件

MindSpore Transformers中已提供DeepSeek-V3基于MindSpore的实现，主要涉及的文件有：

1. 模型具体实现：

    ```text
     deepseek3/
      ├── deepseek2_model.py                  # DeepSeek-V2模型代码
      ├── deepseek2_config.py                 # DeepSeek-V2配置代码
      ├── deepseek3_model.py                  # DeepSeek-V3模型代码
      └── deepseek3_config.py                 # DeepSeek-V3配置代码
    ```

2. 模型配置：

    ```text
     deepseek3/
      ├── parallel_speed_up.json              # 数据集并行通信配置
      └── deepseek3_671b/
           └── pretrain_deepseek3_671b.yaml   # 预训练任务配置
    ```

3. 数据集处理脚本：

    ```text
    deepseek3/
     └── wikitext_to_bin.py           # wikitext数据预处理
    ```

## 预训练

MindSpore Transformers支持对DeepSeek-V3进行预训练。仓库中提供了一份[预训练配置文件](#模型文件)供参考，该配置基于128台Atlas 800T A2（64G），使用Wikitext-2数据集进行预训练。为了方便体验，本章节基于此配置进行修改，缩小了DeepSeek-V3模型参数量，使其能够在单台Atlas 800T A2（64G）上拉起预训练流程。

### 环境准备

准备一台Atlas 800T A2（64G）训练服务器。MindSpore Transformers的环境依赖如下：

| Python | MindSpore |     CANN      |  固件与驱动   |
|:------:|:---------:|:-------------:|:--------:|
|  3.10  |  2.4.10   | 8.0.RC3.beta1 | 24.1.RC3 |

#### 安装固件与驱动

点击[此处](https://www.hiascend.com/hardware/firmware-drivers/community)下载固件与驱动的安装包，参考[昇腾官方教程](https://www.hiascend.com/document/detail/zh/quick-installation/24.0.RC1/quickinstg_train/800_9000A2/quickinstg_800_9000A2_0007.html)进行安装。

> 固件与驱动需要安装24.1.RC3及以上版本，版本过低请进行升级

#### 准备Docker容器：

提供了DeepSeek-V3预训练专用Docker镜像（镜像中已包含CANN、MindSpore，无需手动安装），通过如下步骤进行软件环境搭建。

1. 下载Docker镜像

   使用如下命令下载DeepSeek-V3预训练专用镜像：

   ```bash
   docker pull swr.cn-central-221.ovaijisuan.com/mindformers/deepseek_v3_mindspore2.4.10-train:20250209
   ```

2. 基于镜像创建容器

   使用如下命令新建容器：

   ```bash
   image_name=swr.cn-central-221.ovaijisuan.com/mindformers/deepseek_v3_mindspore2.4.10-train:20250209
   docker_name=deepseek_v3
   docker run -itd -u root \
   --ipc=host --net=host \
   --privileged \
   --device=/dev/davinci0 \
   --device=/dev/davinci1 \
   --device=/dev/davinci2 \
   --device=/dev/davinci3 \
   --device=/dev/davinci4 \
   --device=/dev/davinci5 \
   --device=/dev/davinci6 \
   --device=/dev/davinci7 \
   --device=/dev/davinci_manager \
   --device=/dev/devmm_svm \
   --device=/dev/hisi_hdc \
   -v /etc/localtime:/etc/localtime \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
   -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/bin/hccn_tool \
   -v /etc/ascend_install.info:/etc/ascend_install.info \
   -v /var/log/npu:/usr/slog \
   -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
   -v /etc/hccn.conf:/etc/hccn.conf \
   --name "$docker_name" \
   "$image_name" \
   /bin/bash
   ```

3. 进入容器

   使用如下命令进入容器，并进入代码目录：

   ```bash
   docker exec -ti deepseek_v3 bash
   export MINDFORMERS_HOME=/home/work/mindformers
   cd $MINDFORMERS_HOME
   ```

#### 安装MindSpore Transformers

> 镜像中已经安装好了DeepSeek-V3预训练所需的MindSpore Transformers版本，如使用镜像可跳过此步骤。

执行如下命令拉取MindSpore Transformers代码，并编译安装：

```shell
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformers
git checkout e45eb7c5
bash build.sh
```

### 数据集准备

以Wikitext-2数据集为例，参考如下步骤将数据集处理成Megatron BIN格式文件。

1. 下载数据集和分词模型文件

   - 数据集下载：[WikiText2数据集](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip)

   - 分词模型下载：分词模型[tokenizer.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json?download=true)

2. 生成Megatron BIN格式文件

   将数据集文件`wiki.train.tokens`和分词模型文件`tokenizer.json`放置在`../dataset`下

   使用以下命令将数据集文件转换为BIN格式文件。

   ```shell
   cd $MINDFORMERS_HOME
   python research/deepseek3/wikitext_to_bin.py \
   --input ../dataset/wiki.train.tokens \
   --output-prefix ../dataset/wiki_4096 \
   --vocab-file ../dataset/tokenizer.json \
   --seq-length 4096 \
   --worker 1
   ```

3. 构建Megatron BIN数据集模块

   执行如下命令构建Megatron BIN数据集模块。如使用提供的镜像请跳过此操作。

   ```shell
   pip install pybind11
   cd $MINDFORMERS_HOME/mindformers/dataset/blended_datasets
   make
   ```

### 修改配置

修改预训练配置文件`pretrain_deepseek3_671b.yaml`，使其能够在单台Atlas 800T A2（64G）上运行，保存为`pretrain_deepseek3_1b.yaml`。

1. 修改模型配置

   按照如下方式修改以缩小模型规模：

   ```yaml
   # model config
   model:
     model_config:
       type: DeepseekV3Config
       auto_register: deepseek3_config.DeepseekV3Config
       seq_length: 4096
       hidden_size: 2048                                 # 修改为2048
       num_layers: &num_layers 3                         # 修改为3
       num_heads: 8                                      # 修改为8
       max_position_embeddings: 4096
       intermediate_size: 6144                           # 修改为6144
       kv_lora_rank: 512
       n_kv_heads: 128
       q_lora_rank: 1536
       qk_rope_head_dim: 64
       v_head_dim: 128
       qk_nope_head_dim: 128
       vocab_size: 129280
       multiple_of: 256
       rms_norm_eps: 1.0e-6
       bos_token_id: 100000
       eos_token_id: 100001
       pad_token_id: 100001
       ignore_token_id: -100
       compute_dtype: "bfloat16"
       layernorm_compute_type: "float32"
       softmax_compute_type: "float32"
       rotary_dtype: "float32"
       router_dense_type: "float32"
       param_init_type: "float32"
       use_past: False
       extend_method: "None"
       use_flash_attention: True
       offset: 0                                         # 修改为0
       checkpoint_name_or_path: ""
       theta: 10000.0
       return_extra_loss: False
       mtp_depth: &mtp_depth 1
       mtp_loss_factor: 0.3
     arch:
       type: DeepseekV3ForCausalLM
       auto_register: deepseek3_model.DeepseekV3ForCausalLM
   ```

2. 修改MoE配置

   按照如下方式修改以缩小专家混合结构的规模：

   ```yaml
   #moe
   moe_config:
     expert_num: &expert_num 16                          # 修改为16
     expert_group_size: 8
     capacity_factor: 1.5
     aux_loss_factor: 0.05
     num_experts_chosen: 8
     routing_policy: "TopkRouterV2"
     enable_sdrop: False
     balance_via_topk_bias: &balance_via_topk_bias True
     topk_bias_update_rate: &topk_bias_update_rate 0.0001
     use_fused_ops_topkrouter: True
     group_wise_a2a: False
     shared_expert_num: 1
     routed_scaling_factor: 2.5
     norm_topk_prob: False
     first_k_dense_replace: 1                            # 修改为1
     moe_intermediate_size: 2048
     topk_group: 4
     n_group: 8
     aux_loss_factors: [0.001, 0., 0.]
     aux_loss_types: ["expert", "device", "comm"]
     z_loss_factor: 0.0
     expert_model_parallel: 1
     use_gating_sigmoid: True
   ```

3. 修改并行配置

   缩小每种并行方式的切分数目，以适合在单台Atlas 800T A2（64G）上并行训练：

   ```yaml
   # parallel config for devices num=8
   parallel_config:
     data_parallel: &dp 2                                    # 修改为2
     model_parallel: 2                                   # 修改为2
     pipeline_stage: 2                                   # 修改为2
     expert_parallel: 2                                  # 修改为2
     micro_batch_num: &micro_batch_num 4                 # 修改为4
     vocab_emb_dp: True
     use_seq_parallel: True
     gradient_aggregation_group: 4

   # parallel context config
   parallel:
     parallel_mode: 1 # 0-data parallel, 1-semi-auto parallel, 2-auto parallel, 3-hybrid parallel
     gradients_mean: False
     enable_alltoall: True
     full_batch: False
     dataset_strategy: [[*dp, 1], [*dp, 1], [*dp, 1], [*dp, 1]]
     search_mode: "sharding_propagation"
     enable_parallel_optimizer: True
     strategy_ckpt_save_file: "./ckpt_strategy.ckpt"
     parallel_optimizer_config:
       gradient_accumulation_shard: False
       parallel_optimizer_threshold: 64
       optimizer_weight_shard_size: 8                    # 修改为8
   ```

4. 修改学习率配置

   由于Wikitext-2数据集比较小，所以需要缩小学习率预热步数：

   ```yaml
   # lr schedule
   lr_schedule:
     type: ConstantWarmUpLR
     learning_rate: 2.2e-4
     warmup_steps: 20                                    # 修改为20
     total_steps: -1
   ```

5. 修改数据集配置

   配置数据集BIN文件路径：

   ```yaml
   # dataset
   train_dataset: &train_dataset
     data_loader:
       type: BlendedMegatronDatasetDataLoader
       datasets_type: "GPTDataset"
       sizes:
         - 1000
         - 0
         - 0
       config:
         random_seed: 1234
         seq_length: 4096
         split: "1, 0, 0"
         reset_position_ids: False
         reset_attention_mask: False
         eod_mask_loss: False
         num_dataset_builder_threads: 1
         create_attention_mask: False
         data_path:
           - 1
           - "../dataset/wiki_4096_text_document"              # 修改此项为数据集BIN文件路径
       shuffle: False
     input_columns: ["input_ids", "labels", "loss_mask", "position_ids"]
     construct_args_key: ["input_ids", "labels"]
     num_parallel_workers: 8
     python_multiprocessing: False
     drop_remainder: True
     repeat: 1
     numa_enable: False
     prefetch_size: 1
   train_dataset_task:
     type: CausalLanguageModelDataset
     dataset_config: *train_dataset
   ```

   配置数据集并行通信配置路径：

   ```yaml
   # mindspore context init config
   context:
     mode: 0 #0--Graph Mode; 1--Pynative Mode
     device_target: "Ascend"
     max_call_depth: 10000
     max_device_memory: "55GB"
     save_graphs: False
     save_graphs_path: "./graph"
     jit_config:
       jit_level: "O1"
     ascend_config:
       parallel_speed_up_json_path: "./research/deepseek3/parallel_speed_up.json"  # 修改此项为数据集并行通信配置路径
   ```

### 拉起任务

进入DeepSeek-V3代码目录并执行以下命令拉起单台Atlas 800T A2（64G）预训练任务：

```shell
export MS_DEV_DYNAMIC_SINK1=False
cd $MINDFORMERS_HOME
bash scripts/msrun_launcher.sh "run_mindformer.py \
--register_path research/deepseek3 \
--config research/deepseek3/deepseek3_671b/pretrain_deepseek3_1b.yaml"
```

上述命令执行完毕后，训练任务将在后台执行，过程日志保存在`./output/msrun_log`下，使用以下命令可查看训练状态（由于开启了流水并行`pipeline_stage: 2`，真实loss只显示在最后一张卡的日志`worker_7.log`中，其余卡均显示`loss`为`0`）：

```shell
tail -f ./output/msrun_log/worker_7.log
```

训练过程中的权重checkpoint将会保存在`./output/checkpoint`下。

### 扩展：多机训练

如果服务器资源充足，可以参考如下方式拉起多台Atlas 800T A2（64G）训练。

在每台服务器上执行如下命令。设置`master_ip`为主节点IP地址，即`Rank 0`服务器的IP；`node_rank`为每个节点的Rank序号，从`0`到`1023`。

```shell
master_ip=192.168.1.1
node_rank=0

cd $MINDFORMERS_HOME
export MS_DEV_DYNAMIC_SINK1=False
bash scripts/msrun_launcher.sh "run_mindformer.py \
--register_path research/deepseek3 \
--config research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml" \
1024 8 $master_ip 8118 $node_rank output/msrun_log False 7200
```

> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号。

如有关于DeepSeek-V3预训练的相关问题，可以在MindSpore Transformers的Gitee仓库中[提交ISSUE](https://gitee.com/mindspore/mindformers/issues/new)以获取支持。