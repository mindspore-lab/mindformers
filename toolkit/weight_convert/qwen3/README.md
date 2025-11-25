# Qwen3 权重转换脚本说明

MindSpore Transformers 提供了对 Qwen3 模型从 MindSpore Transformers 训练权重反转到 Hugging Face 权重的离线转换脚本，如需加载 HuggingFace 权重进行训练，可以直接在训练 yaml 文件中配置 `pretrained_model_dir` 为 Hugging Face 权重路径（包括模型配套的 config.json 等配置文件及词表），同时设置 `auto_trans_ckpt: True` 即可在线加载 Hugging Face 权重，方便进行微调等任务。

## MindSpore Transformers 权重反转为 Hugging Face 权重

本脚本适用于，将 MindSpore Transformers 训练后得到的权重反转为 HuggingFace 格式的权重，便于进行社区发布或者 vLLM 推理等任务。

> 注：
> 1. 进行权重反转前，需要对训练权重进行去优化器合并；
> 2. 反转脚本的各参数值需要与训练时的 yaml 中配置进行对齐。

首先对训练权重进行[去优化器合并](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/safetensors.html#%E6%9D%83%E9%87%8D%E5%88%87%E5%88%86%E4%B8%8E%E5%90%88%E5%B9%B6)，此处假设训练了 1000 步，权重保存时开启了去冗余保存，训练任务的策略文件夹在 `output/strategy` 下，权重在 `output/checkpoint` 下。

假设此处得到合并后权重的路径为 `MS_TRAIN_CKPT_PATH`，权重去冗余合并指令为：

```bash
python toolkit/safetensors/unified_safetensors.py \
  --src_strategy_dirs "output/strategy" \
  --mindspore_ckpt_dir "output/checkpoint" \
  --output_dir "/path/to/unified_train_ckpt" \
  --file_suffix "1000_1" \
  --filter_out_param_prefix "adam_" \
  --has_redundancy False
```

| 配置项                     | 数据类型   | 是否可选 | 默认值   | 说明                                                                                                           |
|-------------------------|--------|------|-------|--------------------------------------------------------------------------------------------------------------|
| src_strategy_dirs       | string | 必选   | 无     | 训练时 MindSpore Transformers 的策略文件路径，一般存在 `output/strategy` 文件夹底下。                                             |
| mindspore_ckpt_dir      | string | 必选   | 无     | 训练时保存的 MindSpore Transformers 训练权重路径，下面存在若干个 "rank_x" 文件夹，存有各卡训练时存下的权重文件。                                    |
| output_dir              | string | 必选   | 无     | 合并后的 MindSpore Transformers 训练权重路径。                                                                          |
| file_suffix             | string | 必选   | 无     | 合并的权重训练步数的前缀，如训练的第 1000 步各卡文件名若为 `qwen3_rank_x-1000_1.safetensors`，则此处配置为 `"1000_1"`，表示合并时取文件名中含有该前缀的权重进行合并。 |
| filter_out_param_prefix | string | 必选   | 无     | 去优化器合并时，需要配置此项，且配置为 `"adam_"`。                                                                               |
| has_redundancy          | bool   | 可选   | True  | 需要合并的权重是否有冗余，若为去冗余保存，则需将此处设置为 `False`。                                                                       |

上述指令权重合并完后，权重会保存在 `'/path/to/unified_train_ckpt'` 文件夹底下的 `'1000_1_ckpt_convert/unified_safe'` 文件夹中。

假设下面使用 `MS_TRAIN_CKPT_PATH` 代指 `'1000_1_ckpt_convert/unified_safe'`，则可以使用反转脚本将权重反转为 Hugging Face 格式。以 Qwen3-0.6B 参数为例，反转脚本的使用示例指令如下：

```bash
python convert_weight.py \
  --input_path MS_TRAIN_CKPT_PATH \
  --output_path HF_REVERSE_CKPT_PATH \
  --model 'qwen3' \
  --reversed \
  --num_layers 28 \
  --num_attention_heads 16 \
  --num_query_groups 8 \
  --kv_channels 128 \
  --ffn_hidden_size 3072 \
  --dtype 'bf16' \
  --max_worker 16
```

所有指令参数介绍如下：

> 注：指令参数的默认值为 Qwen3-0.6B 模型的参数量，若进行其他参数量（如8B、32B）的模型训练时，需要将如下参数与 yaml 中相应配置进行对齐，方可进行转换。

| 配置项                 | 数据类型   | 是否可选 | 默认值    | 说明                                                                                                                |
|---------------------|--------|------|--------|-------------------------------------------------------------------------------------------------------------------|
| input_path          | string | 必选   | 无      | 需要转换的 MindSpore Transformers 训练权重路径。                                                                              |
| output_path         | string | 必选   | 无      | 转换后的 Hugging Face 权重的目标路径。                                                                                        |
| model               | string | 必选   | 无      | 选择进行权重转换的模型，此处对应配置为 `'qwen3'`。                                                                                    |
| reversed            | bool   | 必选   | False  | 是否进行权重反转。使用时，仅需设置 `--reversed` 即可，效果与 `--reversed True` 一致。                                                       |
| num_layers          | int    | 可选   | 28     | 模型层数，对应训练 yaml 文件中的 `model.mocel_config.num_layers`（别名可能为`model.mocel_config.num_hidden_layers`）。                 |
| num_attention_heads | int    | 可选   | 16     | Transformer 注意力头数，对应训练 yaml 文件中的 `model.mocel_config.num_attention_heads`。                                        |
| num_query_groups    | int    | 可选   | 8      | 组查询注意力的查询组数量，对应训练 yaml 文件中的 `model.mocel_config.num_query_groups`（别名可能为`model.mocel_config.num_key_value_heads`）。 |
| kv_channels         | int    | 可选   | 128    | 多头注意力中的投影权重维度，对应训练 yaml 文件中的 `model.mocel_config.kv_channels`（别名可能为`model.mocel_config.head_dim`）。                |
| ffn_hidden_size     | int    | 可选   | 3072   | 模型前馈神经网络层的维度，对应训练 yaml 文件中的 `model.mocel_config.ffn_hidden_size`。（别名可能为`model.mocel_config.intermediate_size`）。   |
| dtype               | string | 可选   | 'bf16' | 目标转换的 Hugging Face 权重数据类型，可选为 `'bf16'` 、 `'fp16'` 和 `'fp32'` ，默认为 `'bf16'` 。                                      |
| max_worker          | int    | 可选   | 16     | 使用多少个子进程进行权重处理。请合理控制此项，避免开启过多子进程造成资源竞争，这有可能会导致内存溢出（OOM），默认值为 `16` 。                                               |
