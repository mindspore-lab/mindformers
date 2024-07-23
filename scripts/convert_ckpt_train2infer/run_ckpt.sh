#!/bin/bash

#train2infer
#bash ckpt_convert_v4.sh -t true -p fp16 -w 8 -y /home/checkpoint_download/llama57b/predict_llama2_57b_910b.yaml  -sc /apps/predict/ckpt/57b/train_ckpt/2024-05-20-283200 -ts /apps/predict/ckpt/57b/train_ckpt/0520_strategy/strategy > log_train_infer_fp16_8p.txt 2>&1 &

#fp16_4p to  w8a8_4p
#bash ckpt_convert_v1.sh -t false -p w8a8 -w 4 -y /home/checkpoint_download/llama57b_quant_4p/predict_llama2_57b_910b.yaml -sc ./infer_ckpt/fp16_4p_qkv -dc ./infer_qkv_ckpt  -is ./infer_qkv_strategy > log_fp16tow8a8_4to4_qkv.txt 2>&1 &

#fp16_8p to w8a16_4p
#bash ckpt_convert_v2.sh -t false -p w8a16 -w 4 -y /home/checkpoint_download/llama57b/predict_llama2_57b_910b.yaml -sc ./infer_ckpt/fp16_8p -dc ./infer_qkv_ckpt  -is ./infer_qkv_strategy > log_fp16tow8a16_4to4_qkv.txt 2>&1 &


bash ckpt_convert_v3.sh -t false -p w8a16 -w 4 -y /home/checkpoint_download/llama57b/predict_llama2_57b_910b.yaml -sc /apps/predict/pkgs/0720/test_ckpt/fp16_vp_qkv_8p  > log_fp16tow8a16_8to4.txt 2>&1 &
