#!/bin/bash

# help information
print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -t,  --train_to_infer        Convert checkpoint from training to inference (true: training to inference, false: fp16 infer ckpt to quant infer ckpt)"
    echo "  -p,  --precision             Set precision (fp16, w8a16 or w8a8)"
    echo "  -w,  --world_size            Set the world size for distributed training (2, 4 or 8)"
    echo "  -y,  --yaml_path             Yaml path or model config path"
    echo "  -d,  --dataset_name          Dataset for convert w8a8 weights (boolq, squad1.1 and wikitext2)"
    echo "  -dp, --dataset_path          Dataset path (e.g. './boolq/dev.jsonl' )"
    echo "  -ts, --train_strategy_file   Training strategy saved path"
    echo "  -is, --infer_strategy_file   Inference strategy saved path"
    echo "  -sc, --src_ckpt_path         Source ckpt path"
    echo "  -dc, --dst_ckpt_path         Destination ckpt path"
    echo "  -pm, --pt_to_ms              Pytorch pretrained weights convert to mindspore checkpoint (true or false) "
    echo "  -pp, --pipeline_stage        Pipeline_stage set during training "
    echo "  -h,  --help                  Print this help message"
}

# parser args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--train_to_infer) train_to_infer="$2"; shift ;;
        -p|--precision) precision="$2"; shift ;;
        -w|--world_size) world_size="$2"; shift ;;
        -y|--yaml_path) yaml_path="$2"; shift ;;
        -d|--dataset_name) dataset_name="$2"; shift ;;
        -dp|--dataset_path) dataset_path="$2"; shift ;;
        -ts|--train_strategy_file) train_strategy_file="$2"; shift ;;
        -is|--infer_strategy_file) infer_strategy_file="$2"; shift ;;
        -sc|--src_ckpt_path) src_ckpt_path="$2"; shift ;;
        -dc|--dst_ckpt_path) dst_ckpt_path="$2"; shift ;;
        -pm|--pt_to_ms) pt_to_ms="$2"; shift ;;
        -pp|--pipeline_stage) pipeline_stage="$2"; shift ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "Unknown option: $1"; print_usage; exit 1 ;;
    esac
    shift
done

# check input is not none
if [ -z "$precision" ] || [ -z "$world_size" ]; then
    echo "Error: Missing precision and world_size required options."
    print_usage
    exit 1
fi

# print
echo "Settings:"
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
    if [ -f  ${Infer_strategy_path}_no_qkv/strategy/ckpt_strategy_rank_0.ckpt ] ; then
        echo "Has ${precision}_${world_size}p strategy, jump to step 2"
    else
        msrun --worker_num=$world_size --local_worker_num=$world_size \
        --log_dir=./log/msrun_log_save_strategy_${precision}_${world_size}p \
        --master_port=8126 --join=True --bind_core=False \
        python save_strategy.py \
        --yaml_file=$yaml_path \
        --save_strategy_path=${Infer_strategy_path}_no_qkv \
        --world_size=$world_size \
        --qkv_concat=False > ./log/log_save_strategy_${precision}_${world_size}p.log 2>&1
    fi
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
    --infer_strategy_file=${Infer_strategy_path}_no_qkv/strategy \
    --train_2_infer_path=$Infer_ckpt_path"/train_2_infer_ckpt" \
    --infer_ckpt_path=$Infer_ckpt_path \
    --world_size=$world_size \
    --pipeline_stage=$pipeline_stage > ./log/log_train_2_infer_${world_size}.log 2>&1
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
elif [ "$train_to_infer" == "false" ]; then
    echo "Converting checkpoint from fp16 inference to ${precision} inference"
    # 检查输入值
    if [ -z "$src_ckpt_path" ] || [ -z "$yaml_path" ] ; then
        echo "Please set fp16 checkpoint saved path and ${precision} yaml file"
        exit 1
    fi
    if [ -z "$dst_ckpt_path" ]; then
       Dst_ckpt_path=./infer_ckpt/${precision}
    else
       Dst_ckpt_path=${dst_ckpt_path}/${precision}
    fi
    if [ -z "$infer_strategy_file" ]; then
       Infer_strategy_path=./infer_strategy/${precision}
    else
       Infer_strategy_path=${infer_strategy_file}/${precision}
    fi
    # 检查是否是相同卡数
    dir_count=$(find  "$src_ckpt_path"  -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "Number of directories in '$src_ckpt_path' is: $dir_count"
    if [ "$dir_count" != "$world_size" ] ; then
        if find "${Dst_ckpt_path}_${dir_count}p/rank_0/" -type f -name "*.ckpt" | read; then
            echo "Has ${precision}_${dir_count}p ckpt, jump to step 2"
        else
            if [ "$precision" == "w8a16" ]; then
                #1. 转换成dir_count的w8a16权重
                echo "-----1. Start to convert ${dir_count}p ${precision} weights time: $(date +%H:%M:%S) -----"
                msrun --worker_num=$dir_count --local_worker_num=$dir_count --master_port=8126 \
                --log_dir=./log/msrun_log_w8a16_${dir_count}_ckpt --join=True --cluster_time_out=300 \
                python llama2_w8a16_quant_ckpt.py \
                -c $yaml_path \
                -l $src_ckpt_path \
                -o ${Dst_ckpt_path}_${dir_count}p \
                -w $dir_count \
                -a $precision \
                > ./log/log_fp16_to_${precision}_${dir_count}.log 2>&1
                if find "${Dst_ckpt_path}_${dir_count}p/w8a16_ckpt/rank_0/" -type f -name "*.ckpt" | read; then
                    echo "-----1. End convert ${dir_count}p ${precision} weights time: $(date +%H:%M:%S) -----"
                    mv ${Dst_ckpt_path}_${dir_count}p/w8a16_ckpt/*  ${Dst_ckpt_path}_${dir_count}p/
                    rm -rf ${Dst_ckpt_path}_${dir_count}p/w8a16_ckpt
                else
                    echo "ERROR"
                    exit 1
                fi
            elif [ "$precision" == "w8a8" ]; then
                #1. 转换成dir_count的w8a8权重
                echo "-----1. Start to convert ${dir_count}p ${precision} weights time: $(date +%H:%M:%S) -----"
                msrun --worker_num=$dir_count --local_worker_num=$dir_count --master_port=8126 \
                --log_dir=./log/msrun_log_w8a8_${dir_count}_ckpt --join=True --cluster_time_out=300 \
                python llama2_w8a8_quant_ckpt.py \
                -c $yaml_path \
                -l $src_ckpt_path \
                -o ${Dst_ckpt_path}_${dir_count}p \
                -w $dir_count \
                -s $dataset_path \
                -t $dataset_name \
                > ./log/log_fp16_to_${precision}_${dir_count}.log 2>&1
                if find "${Dst_ckpt_path}_${dir_count}p/w8a8_ckpt/rank_0/" -type f -name "*.ckpt" | read; then
                    echo "-----1. End convert ${dir_count}p ${precision} weights time: $(date +%H:%M:%S) -----"
                    mv ${Dst_ckpt_path}_${dir_count}p/w8a8_ckpt/*  ${Dst_ckpt_path}_${dir_count}p/
                    rm -rf ${Dst_ckpt_path}_${dir_count}p/w8a8_ckpt
                else
                    echo "ERROR"
                    exit 1
                fi
            elif [ "$precision" == "fp16" ]; then
                echo "-----Start to convert ${dir_count}p ${precision} to ${world_size}p ${precision} -----"
            else
                echo "Wrong precision input"
                exit 1
            fi
        fi
        if [ -f  ${Infer_strategy_path}_${dir_count}p/strategy/ckpt_strategy_rank_0.ckpt ] ; then
            echo "Has ${precision}_${dir_count}p strategy, jump to step 3"
        else
            #2. 生成dir_count的strategy
            echo "-----2. Start to generate ${dir_count}p ${precision} strategy time: $(date +%H:%M:%S) -----"
            msrun --worker_num=$dir_count --local_worker_num=$dir_count \
            --log_dir=./log/msrun_log_save_strategy_${precision}_${dir_count}p --master_port=8126 \
            --join=True --bind_core=False \
            python save_strategy.py \
            --yaml_file=$yaml_path \
            --save_strategy_path=${Infer_strategy_path}_${dir_count}p \
            --world_size=$dir_count \
            > ./log/log_save_strategy_${precision}_${dir_count}.log 2>&1
            echo "-----2. End generate ${dir_count}p ${precision} strategy time: $(date +%H:%M:%S) -----"
        fi
        if [ -f ${Infer_strategy_path}_${world_size}p/strategy/ckpt_strategy_rank_0.ckpt ] ; then
            echo "Has ${precision}_${world_size}p strategy, jump to step 4"
        else
            #3. 生成world_size的strategy
            echo "-----3. Start to generate ${world_size}p ${precision} strategy time: $(date +%H:%M:%S) -----"
            msrun --worker_num=$world_size --local_worker_num=$world_size \
            --log_dir=./log/msrun_log_save_strategy_${precision}_${world_size}p --master_port=8126 \
            --join=True --bind_core=False \
            python save_strategy.py \
            --yaml_file=$yaml_path \
            --save_strategy_path=${Infer_strategy_path}_${world_size}p \
            --world_size=$world_size \
            > ./log/log_save_strategy_${precision}_${world_size}.log 2>&1
            echo "-----3. End generate ${world_size}p ${precision} strategy time: $(date +%H:%M:%S) -----"
        fi
        if [ "$precision" == "fp16" ]; then
            #4. 转成worldsize的权重
            echo "-----4. Start to convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
            python transform_ckpt.py \
            --src_ckpt_strategy=${Infer_strategy_path}_${dir_count}p/strategy/ckpt_strategy_rank_0.ckpt \
            --dst_ckpt_strategy=${Infer_strategy_path}_${world_size}p/strategy/ckpt_strategy_rank_0.ckpt \
            --src_ckpt_dir=${src_ckpt_path} \
            --dst_ckpt_dir=${Dst_ckpt_path}_${world_size}p_not_adjust_qkv \
            --prefix="checkpoint_" \
            > ./log/log_transform_ckpt_${precision}_${world_size}.log 2>&1
            echo "-----4. End convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
            #5. 转成worldsize的权重
            echo "-----5. Start to adjust qkv from ${dir_count} to ${world_size} ${precision} time: $(date +%H:%M:%S) -----"
            python adjust_qkv.py \
            --src_ckpt_path=${Dst_ckpt_path}_${world_size}p_not_adjust_qkv \
            --dst_ckpt_path=${Dst_ckpt_path}_${world_size}p \
            --dir_count=${dir_count} \
            --world_size=${world_size} \
            --quant=False \
            > ./log/log_adjust_qkv_${dir_count}_to_${world_size}.log 2>&1
            echo "-----5. End adjust qkv from ${dir_count} to ${world_size} ${precision} time: $(date +%H:%M:%S) -----"
        else
            #4. 转成worldsize的权重
            echo "-----4. Start to convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
            python transform_ckpt.py \
            --src_ckpt_strategy=${Infer_strategy_path}_${dir_count}p/strategy/ckpt_strategy_rank_0.ckpt \
            --dst_ckpt_strategy=${Infer_strategy_path}_${world_size}p/strategy/ckpt_strategy_rank_0.ckpt \
            --src_ckpt_dir=${Dst_ckpt_path}_${dir_count}p \
            --dst_ckpt_dir=${Dst_ckpt_path}_${world_size}p_not_adjust_qkv \
            --prefix="checkpoint_" \
            > ./log/log_transform_ckpt_${precision}_${world_size}.log 2>&1
            echo "-----4. End convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
            #5. 转成worldsize的权重
            echo "-----5. Start to adjust qkv from ${dir_count} to ${world_size} ${precision} time: $(date +%H:%M:%S) -----"
            python adjust_qkv.py \
            --src_ckpt_path=${Dst_ckpt_path}_${world_size}p_not_adjust_qkv \
            --dst_ckpt_path=${Dst_ckpt_path}_${world_size}p \
            --dir_count=${dir_count} \
            --world_size=${world_size} \
            > ./log/log_adjust_qkv_${dir_count}_to_${world_size}.log 2>&1
            echo "-----5. End adjust qkv from ${dir_count} to ${world_size} ${precision} time: $(date +%H:%M:%S) -----"
        fi
        rm -rf ${Dst_ckpt_path}_${world_size}p_not_adjust_qkv
    else
        if [ "$precision" == "w8a16" ]; then
            echo "-----1. Start to convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
            msrun --worker_num=$world_size --local_worker_num=$world_size --master_port=8126 \
            --log_dir=./log/msrun_log_w8a16_${world_size}_ckpt --join=True --cluster_time_out=300 \
            python llama2_w8a16_quant_ckpt.py \
            -c $yaml_path \
            -l $src_ckpt_path  \
            -o ${Dst_ckpt_path}_${world_size}p \
            -w $world_size \
            -a $precision \
            > ./log/log_fp16_to_${precision}_${world_size}.log 2>&1
            echo "-----1. End convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
            mv ${Dst_ckpt_path}_${world_size}p/w8a16_ckpt/*  ${Dst_ckpt_path}_${world_size}p/
            rm -rf ${Dst_ckpt_path}_${world_size}p/w8a16_ckpt
        elif [ "$precision" == "w8a8" ]; then
            echo "-----1. Start to convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
            msrun --worker_num=$world_size --local_worker_num=$world_size --master_port=8126 \
            --log_dir=./log/msrun_log_w8a8_${world_size}_ckpt --join=True --cluster_time_out=300 \
            python llama2_w8a8_quant_ckpt.py \
            -c $yaml_path \
            -l $src_ckpt_path  \
            -o ${Dst_ckpt_path}_${world_size}p \
            -w $world_size \
            -s $dataset_path \
            -t $dataset_name \
            > ./log/log_fp16_to_${precision}_${world_size}.log 2>&1
            echo "-----1. End convert ${world_size}p ${precision} weights time: $(date +%H:%M:%S) -----"
            mv ${Dst_ckpt_path}_${world_size}p/w8a8_ckpt/*  ${Dst_ckpt_path}_${world_size}p/
            rm -rf ${Dst_ckpt_path}_${world_size}p/w8a8_ckpt
        else
            echo "Not support current setting!"
        fi
    fi
elif [ "$pt_to_ms" == "true" ]; then
    echo "Converting pytorch bin to mindspore ckpt"
    if [ "$precision" != "fp16" ]; then
      echo "Only support precision is fp16 in pytorch convert to mindspore "
      precision="fp16"
    fi
    if [ -z "$dst_ckpt_path" ]; then
       Dst_ckpt_path=./pt_to_ms/${precision}
    else
       Dst_ckpt_path=${dst_ckpt_path}/${precision}
    fi
    mkdir -p ${Dst_ckpt_path}_complete_${world_size}/rank_0/
    #1. bin to ckpt
    echo "-----1. Start to convert bin to ckpt time: $(date +%H:%M:%S)-----"
    python convert_pt2ms.py \
    --torch_ckpt_dir $src_ckpt_path \
    --mindspore_ckpt_file ${Dst_ckpt_path}_complete_${world_size}/rank_0/llama57b.ckpt \
    > ./log/log_pt_to_ms.log 2>&1
    echo "-----1. End convert bin to ckpt time: $(date +%H:%M:%S)-----"
    #2. 生成world的strategy
    echo "-----2. Start to generate ${world_size} strategy time: $(date +%H:%M:%S)-----"
    msrun --worker_num=$world_size --local_worker_num=$world_size \
    --log_dir=./log/msrun_log_save_strategy_${precision}_${world_size}p --master_port=8126 --join=True --bind_core=False \
    python save_strategy.py \
    --yaml_file=$yaml_path \
    --save_strategy_path=${Dst_ckpt_path}_strategy_${world_size}p_no_qkv \
    --world_size=$world_size \
    --qkv_concat=False \
    > ./log/log_save_strategy_${precision}_${world_size}.log 2>&1
    echo "-----2. End generate ${world_size} strategy time: $(date +%H:%M:%S)-----"
    #3. 切分权重
    echo "-----3. Start to split ${world_size}p ckpt time: $(date +%H:%M:%S)-----"
    python transform_ckpt.py \
    --dst_ckpt_strategy ${Dst_ckpt_path}_strategy_${world_size}p_no_qkv/strategy/ckpt_strategy_rank_0.ckpt \
    --src_ckpt_dir ${Dst_ckpt_path}_complete_${world_size} \
    --dst_ckpt_dir ${Dst_ckpt_path}_without_qkv_concat_${world_size} \
    --prefix "checkpoint_" \
    > ./log/log_pt_to_ms_transf_to${world_size}.log 2>&1
    echo "-----3. End split ${world_size}p ckpt time: $(date +%H:%M:%S)-----"
    #4. 增加前端融合算子
    echo "-----4. Start to convert qkv time: $(date +%H:%M:%S)-----"
    python convert_qkv_ffn.py \
    --world_size=$world_size \
    --src_ckpt_path=${Dst_ckpt_path}_without_qkv_concat_${world_size} \
    --dst_ckpt_path=${Dst_ckpt_path}_ckpt_${world_size} \
    > ./log/log_qkv_concat_pt_to_ms.log 2>&1
    echo "-----4. End convert qkv time: $(date +%H:%M:%S)-----"
    rm -rf ${Dst_ckpt_path}_without_qkv_concat_${world_size}
    rm -rf ${Dst_ckpt_path}_complete_${world_size}
else
    echo "Your input mode of training to inference conversion is not supported."
fi

echo "Convert finish!"
end_time=$(date +%H:%M:%S)
echo "Total Start Time: $start_time, Total End Time: $end_time"