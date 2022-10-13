# 如何转换Huggingface的t5权重

## T5 Model

### 从HuggingFace的官方中搜索t5-small

下载模型权重。`t5-small`的层数为6层，然后执行下述命令

> python convert_t5_weight.py --layers 6 --torch_path pytorch_model.bin --mindspore_path ./converted_mindspore_t5.ckpt

### 加载T5模型，开始执行训练

在`examples/pretrain/pretrain_t5.sh`中，增加`--load_checkpoint_path`参数。
一个完整的示例如下所示。其中`--device_target="Ascend"`表示下述的命令将会在`Ascend`上面执行训练。

```bash
DEVICE_ID=$1
EPOCH_SIZE=$2
DATA_DIR=$3

python -m transformer.train \
    --config='./transformer/configs/t5/t5_base.yaml' \
    --epoch_size=$EPOCH_SIZE \
    --device_id=$DEVICE_ID \
    --train_data_path=$DATA_DIR \
    --optimizer="adam" \
    --max_seq_length=512 \
    --max_decode_length=512 \
    --parallel_mode="stand_alone" \
    --max_position_embeddings=16 \
    --d_kv=64 \
    --global_batch_size=96 \
    --vocab_size=32128 \
    --hidden_size=512 \
    --intermediate_size=2048 \
    --num_hidden_layers=6 \
    --num_attention_heads=8 \
    --load_checkpoint_path='mindspore_t5_small.ckpt'
    --bucket_boundaries=16 \
    --has_relative_bias=True \
    --device_target="Ascend"
```

## OPT Model

### OPT权重下载和OPT词表下载

从HuggingFace的[官网](https://huggingface.co/facebook/opt-2.7b) 下载`facebook/opt-2.7b`模型权重,记名字为`pytorch_model.bin`。`opt-2.7b`的层数为32层，设置为`--layers 32`，然后执行下述命令
将HuggingFace的权重转换为MindSpore的权重。

```bash
python tools/convert_opt_weight.py --layers 32 --torch_path pytorch_model.bin --mindspore_path ./converted_mindspore_opt.ckpt
```

从HuggingFace的[官网](https://huggingface.co/facebook/opt-2.7b) 下载`facebook/opt-2.7b`对应的词表文件，记为`vocab.json`

### 加载OPT模型，开始执行训练

在`examples/pretrain/pretrain_opt_distributed.sh`中，增加`--load_checkpoint_path`参数，指定转换后的权重的文件路径。
一个完整的示例如下所示。下述的命令将会启动OPT在8卡GPU上面进行训练

```bash
bash examples/pretrain/pretrain_opt_distributed.sh EPOCH_SIZE hostfile DATA_DIR
```

### 使用OPT进行推理

使用转换的权重或者训练完成的权重，用户可以使用下述的命令执行执行单卡2.6B模型OPT模型的推理。

在此脚本中 `--device_target="Ascend"`指定运行设备为`Ascend`，用户可以该值修改为`GPU`。

>注意：在此脚本中，已经默认设置load_checkpoint_path=converted_mindspore_opt.ckpt，vocab_path=vocab.json

如果用户需要自定义文件路径，请在`examples/pretrain/eval_opt.sh`进行修改。

```bash
bash examples/pretrain/eval_opt.sh "who are you?"
```
