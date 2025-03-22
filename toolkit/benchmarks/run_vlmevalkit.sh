#!/bin/bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

USER_CMD="$*"
echo "Original command: $USER_CMD"


REGISTER_PATH=""
DATA=""
MODEL=""
VERBOSE=""
WORK_DIR=""
MODEL_PATH=""

function show_help {
    echo "  --register_path    <path>     Set the register path,eg:/path/qwenvl"
    echo "  --data             <data>     Set the data,eg:MVBench"
    echo "  --model            <model>    Set the model,eg:cogvlm2-image-llama3-chat"
    echo "  --verbose          <log>      Set the output log"
    echo "  --work_dir         <dir>      Set the work dir,eg:/path/cogvlm2-result"
    echo "  --model_path       <path>     Set the model path,include the yaml file"
    echo "  --help                        Show this help message"
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --register_path) REGISTER_PATH="$2"; shift ;; # Set register path parameter
        --data) DATA="$2"; shift ;;                   # Set data parameter
        --model) MODEL="$2"; shift ;;                 # Set model parameter
        --verbose) VERBOSE="--verbose" ;;            # Set verbose parameter
        --work_dir) WORK_DIR="$2"; shift ;;           # Set work_dir parameter
        --model_path) MODEL_PATH="$2"; shift ;;       # Set model_path parameter
        --help)
          show_help
            return 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            return 1
            ;;
    esac
    shift
done


if [[ -n "$REGISTER_PATH" ]]; then
    export REGISTER_PATH=$REGISTER_PATH
    echo "REGISTER_PATH is: $REGISTER_PATH"
fi

FINAL_CMD="python toolkit/benchmarks/eval_with_vlmevalkit.py"
if [[ -n "$DATA" ]]; then
    FINAL_CMD="$FINAL_CMD --data $DATA"
fi
if [[ -n "$MODEL" ]]; then
    FINAL_CMD="$FINAL_CMD --model $MODEL"
fi
if [[ -n "$VERBOSE" ]]; then
    FINAL_CMD="$FINAL_CMD $VERBOSE"
fi
if [[ -n "$WORK_DIR" ]]; then
    FINAL_CMD="$FINAL_CMD --work_dir $WORK_DIR"
fi
if [[ -n "$MODEL_PATH" ]]; then
    FINAL_CMD="$FINAL_CMD --model_path $MODEL_PATH"
fi

echo "Executing command: $FINAL_CMD"
eval "$FINAL_CMD"
