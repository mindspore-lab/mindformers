#!/bin/bash

# set default value
run_mode="single"
#model_name="llama2_7b"
model_name="\"\""
model_path="predict_model"
config_path="predict_llama2_7b.yaml"
predict_data="\"I love Beijing, because\" \"Huawei is a company that\""
device_num=2


# define help func
usage() {
  echo "Usage: bash $0 -m <mode> -n <name> -p <path> -c <config> -i <predict> -d <device> -a <args>"
#  echo "  -f <file>: Specify the input file"
#  echo "  -v: Enable verbose mode"
  exit 1
}

export TIME_RECORD='on'

# parsing parameters
OPTS=$(getopt -o m:n:p:c:i:d:a: --long mode:,name:,path:,config:,predict:,device:,args: -- "$@")

if [ $? -ne 0 ]; then
  usage
fi

eval set -- "$OPTS"
#echo "$OPTS"

while true; do
  case "$1" in
    --mode | -m )
      run_mode="$2"
      shift 2
      ;;
    --name | -n )
      model_name="$2"
      shift 2
      ;;
    --path | -p )
      model_path="$2"
      shift 2
      ;;
    --config | -c )
      config_path="$2"
      shift 2
      ;;
    --predict | -i )
      predict_data="$2"
      shift 2
      ;;
    --device | -d )
      device_num="$2"
      shift 2
      ;;
    --args | -a )
      script_args="$2"
      shift 2
      ;;
    -- )
      shift
      break
      ;;
    * )
      usage
      ;;
  esac
done

#echo "run_parallel: $run_parallel"
#echo "model_name: $model_name"
#echo "model_path: $model_path"
#echo "config_path: $config_path"
#echo "predict_data: $predict_data"
#echo "device_num: $device_num"

# set environment
SCRIPT_PATH=$(realpath "$(dirname "$0")")
MF_ROOT_APTH=$(realpath "$SCRIPT_PATH/../../")
export PYTHONPATH=$MF_ROOT_APTH:$PYTHONPATH

export RUN_MODE='predict'

EXECUTION="$SCRIPT_PATH/run_inference.py \
 --model_name "$model_name" \
 --model_path "$model_path" \
 --config_path "$config_path" \
 --predict_data "$predict_data" \
 "$script_args""
echo $EXECUTION

if [ "$run_mode" = "single" ]; then
  eval "python $EXECUTION"
elif [ "$run_mode" = "parallel" ]; then
  bash "$MF_ROOT_APTH"/scripts/msrun_launcher.sh "$EXECUTION" "$device_num"
else
  echo "Only support 'single' or 'parallel' mode, but got $PARALLEL."
fi
