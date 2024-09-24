#!/bin/bash

#train to fp16_8p
#bash ckpt_convert.sh -f train_to_infer -p fp16 -w 8 -y /home/checkpoint_download/llama57b/predict_llama2_57b_910b.yaml  -sc /apps/predict/ckpt/57b/train_ckpt/2024-05-20-283200 -ts /apps/predict/ckpt/57b/train_ckpt/0520_strategy/strategy -pp 7 > log_train_infer_fp16_8p.txt 2>&1 &

#fp16_8p to  w8a8c8_4p
#bash ckpt_convert.sh -f quant_weight -p w8a8c8 -w 4 -y /home/checkpoint_download/llama57b/predict_llama2_57b_910b.yaml -sc ./infer_ckpt/fp16_8p_qkv -dc ./infer_qkv_ckpt  -is ./infer_qkv_strategy -d boolq -dp ./boolq/dev.jsonl > log_fp16tow8a8c8_8to4_qkv.txt 2>&1 &

#fp16_8p to w8a16_8p
#bash ckpt_convert.sh -f quant_weight -p w8a16 -w 8 -y /home/checkpoint_download/llama57b/predict_llama2_57b_910b.yaml -sc ./infer_ckpt/fp16_8p_qkv -dc ./infer_qkv_ckpt  -is ./infer_qkv_strategy -d boolq -dp ./boolq/dev.jsonl > log_fp16tow8a16_8to4_qkv.txt 2>&1 &

#fp16_8p to fp16_4p
#bash ckpt_convert.sh -f distributed_weight_transfer -p fp16 -w 4 -y /home/checkpoint_download/llama57b/predict_llama2_57b_910b.yaml -sc ./infer_ckpt/fp16_8p_qkv  > log_fp16_8to4_qkv.txt 2>&1 &

#huggingface to fp16_8p
#bash ckpt_convert.sh -f pt_to_ms -p fp16 -w 8 -y /home/checkpoint_download/llama57b/predict_llama2_57b_910b.yaml -sc /apps/predict/0729_t2i/actual > log_pt2ms.txt 2>&1  &