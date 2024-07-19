#!/bin/bash

# help information
print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -t,  --train_to_infer        Convert checkpoint from training to inference (true: training to inference, false: fp16 infer ckpt to quant infer ckpt)"
    echo "  -p,  --precision             Set precision (fp16, w8a16 or w8a8)"
    echo "  -w,  --world_size            Set the world size for distributed training (2, 4 or 8)"
    echo "  -y,  --yaml_path             Yaml path or model config path"
    echo "  -b,  --boolq_dataset_path    Boolq dataset path (download from: )"
    echo "  -ts, --train_strategy_file   Training strategy saved path"
    echo "  -is, --infer_strategy_file   Inference strategy saved path"
    echo "  -sc, --src_ckpt_path         Source ckpt path"
    echo "  -dc, --dst_ckpt_path         Destination ckpt path"
    echo "  -pm, --pt_to_ms              Pytorch pretrained weights convert to mindspore checkpoint (true or false) "
    echo "  -h,  --help                  Print this help message"
}

# parser args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--train_to_infer) train_to_infer="$2"; shift ;;
        -p|--precision) precision="$2"; shift ;;
        -w|--world_size) world_size="$2"; shift ;;
        -y|--yaml_path) yaml_path="$2"; shift ;;
        -b|--boolq_dataset_path) boolq_dataset_path="$2"; shift ;;
        -ts|--train_strategy_file) train_strategy_file="$2"; shift ;;
        -is|--infer_strategy_file) infer_strategy_file="$2"; shift ;;
        -sc|--src_ckpt_path) src_ckpt_path="$2"; shift ;;
        -dc|--dst_ckpt_path) dst_ckpt_path="$2"; shift ;;
        -pm|--pt_to_ms) pt_to_ms="$2"; shift ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "Unknown option: $1"; print_usage; exit 1 ;;
    esac
    shift
done

# check input is not none
if [ -z "$train_to_infer" ] || [ -z "$precision" ] || [ -z "$world_size" ]; then
    echo "Error: Missing one or more required options."
    print_usage
    exit 1
fi

# print
echo "Settings:"
echo "  Train to infer: $train_to_infer"
echo "  Precision: $precision"
echo "  World size: $world_size"

export MS_COMPILER_CACHE_ENABLE=0

start_time=$(date +%H:%M:%S)

if [ -e "./log/" ]; then
    echo "./log/ is exit"
else
    mkdir ./log/
fi


# fp16
if [ "$train_to_infer" == "true" ] && [ "$precision" == "fp16" ]; then
    echo "Converting checkpoint from training to inference, selected precision: $precision."
    # fp16 train_2_infer
    if [ -z "$infer_strategy_file" ]; then
       Infer_strategy_path=./infer_strategy/${precision}_${world_size}p
    else
       Infer_strategy_path=${infer_strategy_file}/${precision}_${world_size}p
    fi
    #1. save strategy
    echo "-----1. Start to save strategy time: $(date +%H:%M:%S) -----"
    msrun --worker_num=$world_size --local_worker_num=$world_size \
    --log_dir=./log/msrun_log_save_strategy_${precision}_${world_size}p \
    --master_port=8126 --join=True --bind_core=False \
    python save_strategy.py \
    --yaml_file=$yaml_path \
    --save_strategy_path=$Infer_strategy_path \
    --world_size=$world_size \
    --qkv_concat=False > ./log/log_save_strategy_${precision}_${world_size}p.log 2>&1
    echo "-----1. End save strategy  time: $(date +%H:%M:%S) -----"
    if [ -z "$src_ckpt_path" ] || [ -z "$train_strategy_file" ] ; then
        echo "Please set training checkpoint saved path and training strategy saved path."
        exit 1
    fi
    if [ -z "$dst_ckpt_path" ]; then
       Infer_ckpt_path=./infer_ckpt/${precision}_${world_size}p
    else
       Infer_ckpt_path=${dst_ckpt_path}/${precision}_${world_size}p
    fi
    #2. 训练转推理
    echo "-----2. Start to convert train to infer weights time: $(date +%H:%M:%S) -----"
    python train2infer.py \
    --train_ckpt_path=$src_ckpt_path \
    --del_optim_path=$Infer_ckpt_path"/del_optim" \
    --train_strategy_file=$train_strategy_file \
    --infer_strategy_file=$Infer_strategy_path"/strategy" \
    --train_2_infer_path=$Infer_ckpt_path"/train_2_infer_ckpt" \
    --infer_ckpt_path=$Infer_ckpt_path \
    --world_size=$world_size > ./log/log_train_2_infer_${world_size}.log 2>&1
    echo "-----2. End convert train to infer weights time: $(date +%H:%M:%S) -----"
    #3. 增加前端融合算子
    echo "-----3. Start to convert qkv and ffn time: $(date +%H:%M:%S)-----"
    python convert_qkv_ffn.py \
    --world_size=$world_size \
    --src_ckpt_path=$Infer_ckpt_path \
    --dst_ckpt_path=$Infer_ckpt_path"_qkv" \
    > ./log/log_convert_qkv_ffn.log 2>&1
    echo "-----3. End convert qkv time: $(date +%H:%M:%S)-----"
    rm -rf $Infer_ckpt_path"/del_optim"
    rm -rf $Infer_ckpt_path"/train_2_infer_ckpt"
elif [ "$train_to_infer" == "false" ] && [ "$precision" == "w8a16" ]; then
    echo "Converting checkpoint from fp16 inference to w8a16 inference"
    # 检查输入值
    if [ -z "$src_ckpt_path" ] || [ -z "$yaml_path" ] ; then
        echo "Please set fp16 checkpoint saved path and fp16 yaml file"
        exit 1
    fi
    if [ -z "$dst_ckpt_path" ]; then
       Dst_ckpt_path="./infer_ckpt/"
    else
       Dst_ckpt_path=$dst_ckpt_path"/"
    fi
    if [ -z "$infer_strategy_file" ]; then
       Infer_strategy_path="./infer_strategy/"
    else
       Infer_strategy_path=$infer_strategy_file"/"
    fi
    # 检查是否是相同卡数
    dir_count=$(find  "$src_ckpt_path"  -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "Number of directories in '$src_ckpt_path' is: $dir_count"
    if [ "$dir_count" != "$world_size" ] ; then
        if [[ -e "${Infer_strategy_path}fp16_${dir_count}p/strategy/ckpt_strategy_rank_0.ckpt" ]] ; then
            echo "Has fp16_${dir_count}p strategy, jump to step 2"
        else
            #1. 生成fp16 dir_count的strategy
            echo "-----1. Start to generate ${dir_count}p fp16 strategy time: $(date +%H:%M:%S) -----"
            msrun --worker_num=$dir_count --local_worker_num=$dir_count \
            --log_dir=./log/msrun_log_save_strategy_fp16_${dir_count}p --master_port=8126 --join=True --bind_core=False \
            python save_strategy.py \
            --yaml_file=$yaml_path \
            --save_strategy_path=${Infer_strategy_path}fp16_${dir_count}p \
            --world_size=$dir_count \
            --qkv_concat=False \
            > ./log/log_save_strategy_fp16_${dir_count}.log 2>&1
            echo "-----1. End generate ${dir_count}p fp16 strategy time: $(date +%H:%M:%S) -----"
        fi
        if [[ -e "${Infer_strategy_path}fp16_${world_size}p/strategy/ckpt_strategy_rank_0.ckpt" ]] ; then
            echo "Has fp16_${world_size}p strategy, jump to step 3"
        else
            #2. 生成fp16 world_size的strategy
            echo "-----2. Start to generate ${world_size}p fp16 strategy time: $(date +%H:%M:%S) -----"
            msrun --worker_num=$world_size --local_worker_num=$world_size \
            --log_dir=./log/msrun_log_save_strategy_fp16_${world_size}p --master_port=8126 --join=True --bind_core=False \
            python save_strategy.py \
            --yaml_file=$yaml_path \
            --save_strategy_path=${Infer_strategy_path}fp16_${world_size}p \
            --world_size=$world_size \
            --qkv_concat=False \
            > ./log/log_save_strategy_fp16_${world_size}.log 2>&1
            echo "-----2. End generate ${world_size}p fp16 strategy time: $(date +%H:%M:%S) -----"
        fi
        if [[ -e "${Dst_ckpt_path}fp16_${world_size}p/rank_0/*.ckpt" ]] ; then
            echo "Has fp16_${world_size}p ckpt, jump to step 4"
        else
            #3. 转成worldsize的权重
            echo "-----3. Start to convert ${world_size}p fp16 weights time: $(date +%H:%M:%S) -----"
            python transform_ckpt.py \
            --src_ckpt_strategy=${Infer_strategy_path}fp16_${dir_count}p/strategy/ \
            --dst_ckpt_strategy=${Infer_strategy_path}fp16_${world_size}p/strategy \
            --src_ckpt_dir=$src_ckpt_path \
            --dst_ckpt_dir=${Dst_ckpt_path}fp16_${world_size}p \
            --prefix="checkpoint_" \
            > ./log/log_transform_ckpt_fp16_${world_size}.log 2>&1
            echo "-----3. End convert ${world_size}p fp16 weights time: $(date +%H:%M:%S) -----"
        fi
        if [[ -e "${Dst_ckpt_path}fp16_${world_size}p_qkv/rank_0/*.ckpt" ]]; then
            echo "Has fp16_${world_size}p_qkv ckpt, jump to step 5"
        else
            #4. 生成fp16 world_size的qkv权重
            echo "-----4. Start to convert ${world_size}p fp16 qkv and ffn time: $(date +%H:%M:%S) -----"
            python convert_qkv_ffn.py --world_size=$world_size  --src_ckpt_path=${Dst_ckpt_path}fp16_${world_size}p \
            --dst_ckpt_path=${Dst_ckpt_path}fp16_${world_size}p_qkv > ./log/log_convert_qkv_ffn.log 2>&1
            echo "-----4. End convert ${world_size}p fp16 qkv and ffn time: $(date +%H:%M:%S) -----"
        fi
        #5. 转换成world_size的w8a16权重
        echo "-----5. Start to convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
        msrun --worker_num=$world_size --local_worker_num=$world_size --master_port=8126 \
        --log_dir=./log/msrun_log_w8a16_${world_size}_ckpt --join=True --cluster_time_out=300 \
        python llama2_w8a16_quant_ckpt.py \
        -c $yaml_path \
        -l ${Dst_ckpt_path}fp16_${world_size}p_qkv\
        -o ${Dst_ckpt_path}${precision}_${world_size}p_qkv \
        -w $world_size \
        > ./log/log_fp16_to_${precision}_${world_size}.log 2>&1
        echo "-----5. End convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
        mv ${Dst_ckpt_path}${precision}_${world_size}p_qkv/w8a16_ckpt/*  ${Dst_ckpt_path}${precision}_${world_size}p_qkv/
        rm -rf ${Dst_ckpt_path}${precision}_${world_size}p_qkv/w8a16_ckpt
    else
        if [[ -e "${Dst_ckpt_path}fp16_${world_size}p_qkv/rank_0/*.ckpt" ]]; then
            echo "Has fp16_${world_size}p_qkv ckpt, jump to step 2"
        else
          echo "-----1. Start to convert ${world_size}p fp16 qkv and ffn time: $(date +%H:%M:%S) -----"
          python convert_qkv_ffn.py --world_size=$world_size  --src_ckpt_path=$src_ckpt_path \
          --dst_ckpt_path=${Dst_ckpt_path}fp16_${world_size}p_qkv > ./log/log_convert_qkv_ffn.log 2>&1
          echo "-----1. End convert ${world_size}p ${precision} qkv and ffn time: $(date +%H:%M:%S) -----"
        fi
        echo "-----2. Start to convert ${world_size}p fp16 weights time: $(date +%H:%M:%S) -----"
        msrun --worker_num=$world_size --local_worker_num=$world_size --master_port=8126 \
        --log_dir=./log/msrun_log_w8a16_${world_size}_ckpt --join=True --cluster_time_out=300 \
        python llama2_w8a16_quant_ckpt.py \
        -c $yaml_path \
        -l ${Dst_ckpt_path}fp16_${world_size}p_qkv  \
        -o ${Dst_ckpt_path}${precision}_${world_size}p_qkv \
        -w $world_size \
        > ./log/log_fp16_to_${precision}_${world_size}.log 2>&1
        mv ${Dst_ckpt_path}${precision}_${world_size}p_qkv/w8a16_ckpt/*  ${Dst_ckpt_path}${precision}_${world_size}p_qkv/
        rm -rf ${Dst_ckpt_path}${precision}_${world_size}p_qkv/w8a16_ckpt
        echo "-----2. End convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
    fi
elif [ "$train_to_infer" == "false" ] && [ "$precision" == "w8a8" ]; then
    echo "Converting checkpoint from fp16 inference to w8a8 inference"
    # 检查输入值
    if [ -z "$src_ckpt_path" ] || [ -z "$yaml_path" ] ; then
        echo "Please set fp16 checkpoint saved path and fp16 yaml file"
        exit 1
    fi
    if [ -z "$dst_ckpt_path" ]; then
       Dst_ckpt_path="./infer_ckpt/"
    else
       Dst_ckpt_path=$dst_ckpt_path"/"
    fi
    if [ -z "$infer_strategy_file" ]; then
       Infer_strategy_path="./infer_strategy/"
    else
       Infer_strategy_path=$infer_strategy_file"/"
    fi
    # 检查是否是相同卡数
    dir_count=$(find  "$src_ckpt_path"  -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "Number of directories in '$src_ckpt_path' is: $dir_count"
    if [ "$dir_count" != "$world_size" ] ; then
        if [[ -e "${Infer_strategy_path}fp16_${dir_count}p/strategy/ckpt_strategy_rank_0.ckpt" ]]; then
            echo "Has fp16_${dir_count}p strategy, jump to step 2"
        else
            #1. 生成fp16 dir_count的strategy
            echo "-----1. Start to generate ${dir_count}p fp16 strategy time: $(date +%H:%M:%S) -----"
            msrun --worker_num=$dir_count --local_worker_num=$dir_count \
            --log_dir=./log/msrun_log_save_strategy_fp16_${dir_count}p --master_port=8126 --join=True --bind_core=False \
            python save_strategy.py \
            --yaml_file=$yaml_path \
            --save_strategy_path=${Infer_strategy_path}fp16_${dir_count}p \
            --world_size=$dir_count \
            --qkv_concat=False \
            > ./log/log_save_strategy_fp16_${dir_count}.log 2>&1
            echo "-----1. End generate ${dir_count}p fp16 strategy time: $(date +%H:%M:%S) -----"
        fi
        if [[ -e "${Infer_strategy_path}fp16_${world_size}p/strategy/ckpt_strategy_rank_0.ckpt" ]]; then
            echo "Has fp16_${world_size}p strategy, jump to step 3"
        else
            #2. 生成fp16 world_size的strategy
            echo "-----2. Start to generate ${world_size}p fp16 strategy time: $(date +%H:%M:%S) -----"
            msrun --worker_num=$world_size --local_worker_num=$world_size \
            --log_dir=./log/msrun_log_save_strategy_fp16_${world_size}p --master_port=8126 --join=True --bind_core=False \
            python save_strategy.py \
            --yaml_file=$yaml_path \
            --save_strategy_path=${Infer_strategy_path}fp16_${world_size}p \
            --world_size=$world_size \
            --qkv_concat=False \
            > ./log/log_save_strategy_fp16_${world_size}.log 2>&1
            echo "-----2. End generate ${world_size}p fp16 strategy time: $(date +%H:%M:%S) -----"
        fi
        if [[ -e "${Dst_ckpt_path}fp16_${world_size}p/rank_0/*.ckpt" ]]; then
            echo "Has fp16_${world_size}p ckpt, jump to step 4"
        else
            #3. 转成worldsize的权重
            echo "-----3. Start to convert ${world_size}p fp16 weights time: $(date +%H:%M:%S) -----"
            python transform_ckpt.py \
            --src_ckpt_strategy=${Infer_strategy_path}fp16_${dir_count}p/strategy/ \
            --dst_ckpt_strategy=${Infer_strategy_path}fp16_${world_size}p/strategy \
            --src_ckpt_dir=$src_ckpt_path \
            --dst_ckpt_dir=${Dst_ckpt_path}fp16_${world_size}p \
            --prefix="checkpoint_" \
            > ./log/log_transform_ckpt_fp16_${world_size}.log 2>&1
            echo "-----3. End convert ${world_size}p fp16 weights time: $(date +%H:%M:%S) -----"
        fi
        if [[ -e "${Dst_ckpt_path}fp16_${world_size}p_qkv/rank_0/*.ckpt" ]]; then
            echo "Has fp16_${world_size}p_qkv ckpt, jump to step 5"
        else
            #4. 生成fp16 world_size的qkv权重
            echo "-----4. Start to convert ${world_size}p fp16 qkv and ffn time: $(date +%H:%M:%S) -----"
            python convert_qkv_ffn.py --world_size=$world_size  --src_ckpt_path=${Dst_ckpt_path}fp16_${world_size}p \
            --dst_ckpt_path=${Dst_ckpt_path}fp16_${world_size}p_qkv > ./log/log_convert_qkv_ffn.log 2>&1
            echo "-----4. End convert ${world_size}p fp16 qkv and ffn time: $(date +%H:%M:%S) -----"
        fi
        if [ -z "$boolq_dataset_path" ]; then
            boolq_dataset_path="./boolq/dev.jsonl"
        fi
        #5. 转换成world_size的w8a8权重
        echo "-----5. Start to convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
        msrun --worker_num=$world_size --local_worker_num=$world_size --master_port=8126 \
        --log_dir=./log/msrun_log_w8a8_${world_size}_ckpt --join=True --cluster_time_out=300 \
        python llama2_w8a8_quant_ckpt.py \
        -c $yaml_path \
        -l ${Dst_ckpt_path}fp16_${world_size}p_qkv \
        -o ${Dst_ckpt_path}${precision}_${world_size}p_qkv \
        -w $world_size \
        -s $boolq_dataset_path \
        -t boolq \
        > ./log/log_fp16_to_${precision}_${world_size}.log 2>&1
        echo "-----5. End convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
        mv ${Dst_ckpt_path}${precision}_${world_size}p_qkv/w8a8_ckpt/*  ${Dst_ckpt_path}${precision}_${world_size}p_qkv/
        rm -rf ${Dst_ckpt_path}${precision}_${world_size}p_qkv/w8a8_ckpt
    else
        if [[ -e "${Dst_ckpt_path}fp16_${world_size}p_qkv/rank_0/*.ckpt" ]]; then
            echo "Has fp16_${world_size}p_qkv ckpt, jump to step 2"
        else
          echo "-----1. Start to convert ${world_size}p fp16 qkv and ffn time: $(date +%H:%M:%S) -----"
          python convert_qkv_ffn.py --world_size=$world_size  --src_ckpt_path=$src_ckpt_path \
          --dst_ckpt_path=${Dst_ckpt_path}fp16_${world_size}p_qkv > ./log/log_convert_qkv_ffn.log 2>&1
          echo "-----1. End convert ${world_size}p ${precision} qkv and ffn time: $(date +%H:%M:%S) -----"
        fi
        if [ -z "$boolq_dataset_path" ]; then
            boolq_dataset_path="./boolq/dev.jsonl"
        fi
        echo "-----2. Start to convert ${world_size}p fp16 weights time: $(date +%H:%M:%S) -----"
        msrun --worker_num=$world_size --local_worker_num=$world_size --master_port=8126 \
        --log_dir=./log/msrun_log_w8a8_${world_size}_ckpt --join=True --cluster_time_out=300 \
        python llama2_w8a8_quant_ckpt.py \
        -c $yaml_path \
        -l ${Dst_ckpt_path}fp16_${world_size}p_qkv  \
        -o ${Dst_ckpt_path}${precision}_${world_size}p_qkv \
        -w $world_size \
        -s $boolq_dataset_path \
        -t boolq \
        > ./log/log_fp16_to_${precision}_${world_size}.log 2>&1
        mv ${Dst_ckpt_path}${precision}_${world_size}p_qkv/w8a8_ckpt/*  ${Dst_ckpt_path}${precision}_${world_size}p_qkv/
        rm -rf ${Dst_ckpt_path}${precision}_${world_size}p_qkv/w8a8_ckpt
        echo "-----2. End convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
    fi
elif [ "$pt_to_ms" == "true" ]; then
    echo "Converting pytorch bin to mindspore ckpt"
    # 参数说明
    #input_path: huggingface权重保存目录路径
    #output_path: 权重保存文件名，可以指定自定义保存路径
    if [ "$precision" != "fp16" ]; then
      echo "Only support precision is fp16 in pytorch convert to mindspore "
      precision="fp16"
    fi
    if [ -e "./pt_to_ms" ]; then
      echo "./pt_to_ms is exit"
    else
      mkdir ./pt_to_ms
    fi
    #1. bin to ckpt
    echo "-----1. Start to convert bin to ckpt-----"
    python convert_weight.py \
    --model llama \
    --input_path $src_ckpt_path \
    --output_path ./pt_to_ms/complete_${world_size}/rank_0 \
    > ./log/log_pt_to_ms.log 2>&1
    echo "-----1. End convert bin to ckpt-----"
    #2. 生成world的strategy
    echo "-----2. Start to generate ${world_size} strategy-----"
    msrun --worker_num=$world_size --local_worker_num=$world_size \
    --log_dir=./log/msrun_log_save_strategy_${precision}_${world_size}p --master_port=8126 --join=True --bind_core=False \
    python save_strategy.py \
    --yaml_file=$yaml_path \
    --save_strategy_path=./pt_to_ms/stratrgy_fp16_${world_size}p \
    --world_size=$world_size \
    > ./log/log_save_strategy_${precision}_${world_size}.log 2>&1
    echo "-----2. End generate ${world_size} strategy-----"
    #3. 切分权重
    echo "-----3. Start to split ${world_size}p ckpt-----"
    python transform_ckpt.py \
    --dst_ckpt_strategy ./pt_to_ms/stratrgy_fp16_${world_size}p/strategy \
    --src_ckpt_dir ./pt_to_ms/complete_${world_size} \
    --dst_ckpt_dir ./pt_to_ms/without_qkv_concat_${world_size} \
    --prefix "checkpoint_" \
    > ./log/log_pt_to_ms${world_size}.log 2>&1
    echo "-----3. End split ${world_size}p ckpt-----"
    #4. 增加前端融合算子
    echo "-----4. Start to convert qkv -----"
    python convert_weight.py \
    --qkv_concat=True \
    --w2_transb=True \
    --pre_ckpt_path=./pt_to_ms/without_qkv_concat_${world_size} \
    --mindspore_ckpt_path=./pt_to_ms/ckpt_${world_size} \
    > ./log/log_qkv_concat_pt_to_ms.log 2>&1
    echo "-----4. End convert qkv -----"
    rm -rf ./pt_to_ms/without_qkv_concat_${world_size}
    rm -rf ./pt_to_ms/complete_${world_size}
else
    echo "Your input mode of training to inference conversion is not supported."
fi

echo "Convert finish!"
end_time=$(date +%H:%M:%S)
echo "Total Start Time: $start_time, Total End Time: $end_time"
