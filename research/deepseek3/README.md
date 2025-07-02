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
      ├── deepseek2_model.py                        # DeepSeek-V2模型代码
      ├── deepseek2_config.py                       # DeepSeek-V2配置代码
      ├── deepseek3.py                              # DeepSeek-V3模型代码
      ├── deepseek3_model_infer.py                  # DeepSeek-V3推理代码
      └── deepseek3_config.py                       # DeepSeek-V3配置代码
    ```

2. 模型配置：

    ```text
     deepseek3/
      ├── parallel_speed_up.json                     # 数据集并行通信配置
      ├── deepseek3_671b/
      |     ├── predict_deepseek3_671b.yaml          # 推理任务配置
      |     └── predict_deepseek3_671b_w8a8.yaml     # 量化推理任务配置
      └── deepseek3_r1_671b/
            ├── predict_deepseek_r1_671b.yaml        # 推理任务配置
            └── predict_deepseek_r1_671b_w8a8.yaml   # 量化推理任务配置
    ```

3. 权重转换脚本：

    ```text
    deepseek3/
      └── convert_weight.py                           # HuggingFace Safetensors权重转换为MindSpore Safetensors权重
    ```

4. 推理任务脚本：

    ```text
    deepseek3/
      └── run_predict_deepseek.py                     # 执行推理任务
    ```

## 环境安装

DeepSeek-V3所依赖的版本配套如下：

| MindFormers | MindSpore | CANN | 固件与驱动 |
|:-----------:|:---------:|:----:|:-----:|
|    在研版本     |   在研版本    | 在研版本 | 在研版本  |

环境的详细安装指南参考[环境安装指南](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/installation.html)。

## 模型权重下载

### 推理权重准备

用户可以从HuggingFace官方下载推理权重，`tokenizer.json`文件也在链接中下载。

| 模型名称                         |                                     下载链接                                         |                           备注                          |
|:-----------------------------|:---------------------------------------------------------------------------------------:|:------------------------------------------------------:|
| deepseek-ai/DeepSeek-R1-0528      |               [Link](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)                    |                   推理使用                          |
|deepseek-ai/DeepSeek-v3-0324| [link](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)|推理使用|

下载的权重的数据类型是`float8`，需要转换成`bfloat16`使用，转换命令参考如下：

```shell
python research/deepseek3/fp8_cast_bf16.py \
--input-fp8-hf-path path/to/hf_model_fp8_dir/ \
--output-bf16-hf-path path/to/hf_model_bf16_dir/
```

>`path/to/hf_model_bf16_dir/` 可修改为自定义路径，确保该路径有足够的磁盘空间（约 1.4TB）。

## 推理

MindSpore Transformers支持对DeepSeek-V3的推理，目前推理需要4台Atlas 800T A2（64G）机器。

### 环境准备

参考[环境环境](#环境安装)搭建基础环境。完成之后，配置推理相关的环境变量：

```bash
git clone https://gitee.com/mindspore/mindformers.git -b r1.5.0
export MINDFORMERS_HOME=/path/to/mindformers
export PYTHONPATH=$MINDFORMERS_HOME:$PYTHONPATH
export REGISTER_PATH=/path/to/deepseek3
export HCCL_OP_EXPANSION_MODE=AIV
export MS_ENABLE_LCCL=off
export EXPERIMENTAL_KERNEL_LAUNCH_GROUP="thread_num:2,kernel_group_num:8"
cd $MINDFORMERS_HOME
```

环境变量说明：

- MINDFORMERS_HOME：设置为Mindspore Transformers根目录`mindformers/`的绝对路径。
- REGISTER_PATH：设置为DeepSeek-V3模型目录`deepseek3/`的绝对路径。

> 注意事项：

- 使用容器部署推理环境时，如果有部分宿主机的hostname是一致的，需要在起容器的时候修改容器的hostname，保证所有容器的hostname都不一致。

### 权重准备

权重下载参考[推理权重准备](#推理权重准备)，从Hugging Face直接下载模型权重，需要将权重转换为MindSpore的权重，执行以下命令转换：

```bash
python research/deepseek3/convert_weight.py \
--torch_ckpt_path TORCH_CKPT_DIR \
--infer True \
--mindspore_ckpt_path MINDSPORE_CKPT_DIR \
--worker_num 2
```

参数说明：

- torch_ckpt_path：下载HuggingFace权重的文件夹路径。
- infer：是否进行推理权重的转换，默认值：`False`。
- mindspore_ckpt_path：转换后的MindSpore权重文件夹保存路径
- worker_num：多进程转换的进程数，默认值：`4`。

如果使用训练后保存的权重进行推理，需要使用`deepseek3_train2infer.py`脚本将其转换为推理格式。执行以下命令进行转换：

```bash
python research/deepseek3/deepseek3_train2infer.py \
--input_path TRAIN_CKPT_DIR \
--output_path INFER_CKPT_DIR
```

参数说明：

- `input_path`：包含训练权重.safetensors文件的目录路径
- `output_path`：转换后的推理权重保存目录

其他可选参数：

- `first_k_dense_replace`：MoE层之前的稠密层数量，需要与训练时模型的配置一致，默认值：`3`
- `num_heads`：注意力头数，需要与训练时模型的配置一致，默认值：`128`
- `qk_nope_head_dim`：无位置编码的Q/K头维度，需要与训练时模型的配置一致，默认值：`128`
- `qk_rope_head_dim`：带位置编码的Q/K头维度，需要与训练时模型的配置一致，默认值：`64`
- `processes`：并行转换的进程数，默认值：`32`

### 修改配置

仓库上提供的`research/deepseek3/deepseek3_671b/predict_deepseek3_671b.yaml`中有部分配置需要根据实际进行修改，需要修改的地方如下：

`load_checkpoint`需要修改成权重存放的文件夹的绝对路径。

```yaml
load_checkpoint: "/path/to/deepseekv3/model.safetensors"
```

`vocab_file`和`tokenizer_file`需要修改成`tokenizer.json`的绝对路径。

```yaml
processor:
  tokenizer:
    vocab_file: '/path/to/deepseekv3/tokenizer.json'
    tokenizer_file: '/path/to/deepseekv3/tokenizer.json'
```

### 拉起推理任务

分别在4台机器上执行如下命令进行分布式推理，设置master_ip为IP地址，即Node 0服务器的IP。

在Node 0服务器上执行如下命令：

```sh
master_ip=192.168.1.1
bash scripts/msrun_launcher.sh "research/deepseek3/run_predict_deepseek.py \
--config research/deepseek3/deepseek3_671b/predict_deepseek3_671b.yaml \
--input '请介绍一下北京的景点'" \
32 8 $master_ip 8888 0 output/msrun_log False 300
```

在Node 1服务器上执行如下命令：

```sh
master_ip=192.168.1.1
bash scripts/msrun_launcher.sh "research/deepseek3/run_predict_deepseek.py \
--config research/deepseek3/deepseek3_671b/predict_deepseek3_671b.yaml \
--input '请介绍一下北京的景点'" \
32 8 $master_ip 8888 1 output/msrun_log False 300
```

在Node 2服务器上执行如下命令：

```sh
master_ip=192.168.1.1
bash scripts/msrun_launcher.sh "research/deepseek3/run_predict_deepseek.py \
--config research/deepseek3/deepseek3_671b/predict_deepseek3_671b.yaml \
--input '请介绍一下北京的景点'" \
32 8 $master_ip 8888 2 output/msrun_log False 300
```

在Node 3服务器上执行如下命令：

```sh
master_ip=192.168.1.1
bash scripts/msrun_launcher.sh "research/deepseek3/run_predict_deepseek.py \
--config research/deepseek3/deepseek3_671b/predict_deepseek3_671b.yaml \
--input '请介绍一下北京的景点'" \
32 8 $master_ip 8888 3 output/msrun_log False 300
```

预期的推理结果如下：

```txt
<｜begin▁of▁sentence｜>请介绍一下北京的景点\n\n北京，作为中国的首都，拥有丰富的历史文化遗产和众多的旅游景点。以下是一些著名的北京景点介绍：\n\n1. 故宫博物院：位于北京市中心，是中国明清两代的皇家宫殿，也是世界上现存规模最大、保存最为完整的木质结构古建筑群。故宫内收藏有大量珍贵的文物和艺术品，是了解中国古代皇家文化的重要窗口。\n\n2. 天安门广场：位于北京市中心，是世界上最大的城市广场之一。广场北侧是天安门城楼，南侧是人民英雄纪念碑，东侧是国家博物馆，西侧是人民大会堂。天安门广场是中国的象征，也是许多重要历史事件的发生地。\n\n3. 颐和园：位于北京市西北郊，是中国清朝时期的皇家园林，也是中国四大名园之一。颐和园以昆明湖和万寿山为基础，融合了江南园林的精致与北方园林的宏伟，是中国古典园林艺术的杰作。\n\n4. 长城：长城是中国古代的军事防御工程，横跨中国北部和中部地区。北京段的长城包括八达岭、慕田峪、金山岭等，是游客体验长城壮丽风光和了解中国古代军事文化的好去处。\n\n5. 天坛：位于北京市南部，是中国明清两代皇帝祭天祈谷的地方。天坛是中国古代祭天文化的代表，其建筑布局严谨，体现了中国古代的宇宙观和哲学思想。\n\n6. 圆明园：位于北京市西北郊，是清朝时期的皇家园林，曾被誉为“万园之园”。圆明园在1860年的第二次鸦片战争中被英法联军焚毁，现为遗址公园，是了解中国近代历史的重要场所。\n\n7. 北海公园：位于北京市中心，是中国现存最古老、保存最完整的皇家园林之一。北海公园以琼华岛为中心，湖光山色与古建筑相映成趣，是市民休闲娱乐的好去处。\n\n8. 北京奥林匹克公园：位于北京市朝阳区，是2008年北京奥运会的主要场馆区。公园内有鸟巢（国家体育场）、水立方（国家游泳中心）等标志性建筑，是体验现代体育文化和建筑艺术的好地方。\n\n9. 南锣鼓巷：位于北京市东城区，是北京最古老的街区之一，也是北京胡同文化的代表。南锣鼓巷以其独特的胡同风貌和丰富的文化底蕴，吸引了众多游客前来探访。\n\n10. 北京动物园：位于北京市西城区，是中国最早的动物园之一。北京动物园内饲养有众多珍稀动物，是了解中国动物多样性和进行科普教育的好地方。\n\n以上只是北京众多景点中的一部分，北京还有许多其他值得一游的地方，如798艺术区、雍和宫、景山公园等。每个景点都有其独特的历史背景和文化价值，值得游客深入探索。<｜end▁of▁sentence｜>
```

如有关于DeepSeek-V3推理的相关问题，可以在MindSpore Transformers的Gitee仓库中[提交ISSUE](https://gitee.com/mindspore/mindformers/issues/new)以获取支持。