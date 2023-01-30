# StableMoE

## 阶段一：学习路由策略

### 运行指令

```text
bash research/ntlb/examples/pretrain/pretrain_gpt.sh DATA_DIR RANK_TABLE_FILE
```

### 脚本内容

```bash
#!/bin/bash

RANK_SIZE=$1
HOSTFILE=$2
DATASET=$3

mpirun --allow-run-as-root -n $RANK_SIZE \
      --output-filename run_distributed_train_gpt \
python -m research.ntlb.transformer.core_gpt_trainer  \
    --device_num=$RANK_SIZE \
    --train_data_path=$DATASET \
    --seq_length=1024 \
    --global_batch_size=8 \
    --vocab_size=50257 \
    --parallel_mode="semi_auto_parallel" \
    --full_batch=False \
    --checkpoint_prefix="gpt" \
    --routing_policy="stable" \
    --routing_stage="s1" \
    --hidden_size=768 \
    --recompute=True \
    --mp_comm_recompute=False \
    --num_layers=12 \
    --num_heads=12 \
    --data_parallel=2 \
    --model_parallel=1 \
    --expert_parallel=2 \
    --expert_num=2 \
    --per_token_num_experts_chosen=1 \
    --device_target="GPU" > 0128_ntlb_train_gpu_log.txt 2>&1 &
```

## 阶段二：基于学到的稳定路由策略学习

### 运行指令

```text
bash research/ntlb/examples/pretrain/pretrain_gpt.sh DATA_DIR RANK_TABLE_FILE
```

### 脚本内容

```bash
#!/bin/bash

RANK_SIZE=$1
HOSTFILE=$2
DATASET=$3

mpirun --allow-run-as-root -n $RANK_SIZE \
      --output-filename run_distributed_train_gpt \
python -m research.ntlb.transformer.core_gpt_trainer  \
    --device_num=$RANK_SIZE \
    --train_data_path=$DATASET \
    --seq_length=1024 \
    --global_batch_size=8 \
    --vocab_size=50257 \
    --parallel_mode="semi_auto_parallel" \
    --full_batch=False \
    --checkpoint_prefix="gpt" \
    --routing_policy="stable" \
    --routing_stage="s2" \
    --hidden_size=768 \
    --recompute=True \
    --mp_comm_recompute=False \
    --num_layers=12 \
    --num_heads=12 \
    --data_parallel=2 \
    --model_parallel=1 \
    --expert_parallel=2 \
    --expert_num=2 \
    --per_token_num_experts_chosen=1 \
    --device_target="GPU" > 0128_ntlb_train_gpu_log.txt 2>&1 &
```