#!/bin/bash

# testcase 1
bash inference_tool.sh --mode single --name llama2_7b --args "--load_checkpoint /home/data/llama2/llama2_7b.ckpt"

# testcase 2
#bash inference_tool.sh --mode single --name llama2_7b --args "--seq_length 2048 --data_seq_len 1024 --data_batch_size 4 --max_new_tokens 12"

# testcase 3
#bash inference_tool.sh --mode single --config /home/projects/mindformers_dev/configs/llama2/predict_llama2_7b.yaml llama2_7b --args "--load_checkpoint /home/data/llama2/llama2_7b.ckpt"
