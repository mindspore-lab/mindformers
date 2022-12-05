# 通过FFN近似计算实现Transformer网络训练加速 

## 介绍

Transformer是功能强大的神经网络模型。训练Transformer模型通常需要耗费大量的时间和资源。Transformer的计算主要来自attention层和FFN层，目前已经有不少文献专注于对这两部分进行优化或近似计算来提升模型的计算效率，从而降低训练成本。现在需要在GPU和昇腾上实现FFN的近似计算。 

## 软件架构

```text
.
├─ research
   └─ FastFFN 
      ├─ examples
         ├─ pretrain
            ├─ pretrain_gptfast.sh #可执行脚本文件
            └─ pretrain_gptfast_post.sh #可执行脚本文件
         └─ weight_transform.py #模型转换文件
      ├─ transformer
         ├─ model
            ├─ CoRe_Transformer.py
            ├─ gptfast.py
            ├─ layers.py
            ├─ loss.py
            ├─ moe.py
            ├─ op_parallel_config.py
         ├─ gptfast_traner.py
         └─ gptfast_traner_post.py

```

## 快速上手

### 快速FFN 一阶段 运行指令

```bash
bash research/FastFFN/examples/pretrain/pretrain_gptfast.sh
```

### 脚本文件内容

```bash
#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "========================================================================"
echo "Please run the script as: "
echo "bash examples/pretrain/pretrain_gpt.sh  DEVICE_ID EPOCH_SIZE DATA_DIR"
echo "for example: bash examples/pretrain/pretrain_gpt.sh 0 40 /path/zh-wiki/"
echo "========================================================================"
export GLOG_v=3
export DEVICE_ID=$1
EPOCH_SIZE=1
DATA_DIR="/home/ma-user/work/mindspore-fasterFFN/wiki/wikitask/"

echo "========================================================================"
echo "FastFFN GPT 一阶段 "
echo "========================================================================"
python -m research.FastFFN.transformer.gptfast_trainer \
    --epoch_size=$EPOCH_SIZE \
    --train_data_path=$DATA_DIR \
    --optimizer="adam"  \
    --seq_length=1024 \
    --parallel_mode="stand_alone" \
    --checkpoint_prefix="gpt" \
    --global_batch_size=8 \
    --vocab_size=50257 \
    --hidden_size=768 \
    --num_layers=12 \
    --num_heads=12 \
    --full_batch=False \
    --device_target="GPU" > standalone_train_gpu_log.txt 2>&1 &
```

### 权重转换

```bash
python research/FastFFN/examples/weight_transform.py
```

- 修改权重的加载与保存路径

```bash
loadpath = "" # 加载权重的路径
savepath = "" # 保存权重的路径
```

### 快速FFN 二阶段 运行指令

```bash
bash research/FastFFN/examples/pretrain/pretrain_gptfast_post.sh
```

### 脚本文件内容

```bash
#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "================================================================"
echo "Please run the script as: "
echo "bash examples/pretrain/pretrain_gpt.sh  DEVICE_ID EPOCH_SIZE DATA_DIR"
echo "for example: bash examples/pretrain/pretrain_gpt.sh 0 40 /path/zh-wiki/"
echo "================================================================"
export GLOG_v=3
export DEVICE_ID=$1
EPOCH_SIZE=1
DATA_DIR="/home/ma-user/work/mindspore-fasterFFN/wiki/wikitask/"

python -m research.FastFFN.transformer.gptfast_trainer_post \
    --epoch_size=$EPOCH_SIZE \
    --train_data_path=$DATA_DIR \
    --optimizer="adam"  \
    --seq_length=1023 \
    --parallel_mode="stand_alone" \
    --checkpoint_prefix="gpt" \
    --global_batch_size=8 \
    --vocab_size=50257 \
    --hidden_size=768 \
    --num_layers=12 \
    --num_heads=12 \
    --full_batch=False \
    --device_target="GPU" > standalone_train_gpu_log.txt 2>&1 &
```

