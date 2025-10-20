# DeepSeek V3 权重转换脚本说明

MindSpore Transformers 提供了可以从 Hugging Face 权重到 MindSpore Transformers 训练和推理可用权重的相互转换的权重离线转换脚本，下面介绍这两个脚本的使用方法。

## Hugging Face 权重转换为 MindSpore Transformers 权重

本脚本适用于 DeepSeek V3 系列权重（[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)、[DeepSeek-V3-Base](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)、[DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)），以及 DeepSeek-V3.1、DeepSeek-R1 系列权重。

> 注：如果转换从 Hugging Face 官网上下载下来的 DeepSeek 权重，脚本会自动进行 fp8 反量化（默认转换为 bf16 格式），因此转换后的权重文件夹比原始权重文件夹占用磁盘空间要大接近一倍，是正常现象（转换后权重约 1.3T）。

脚本使用示例指令如下：

```bash
python convert_weight.py \
  --input_path HF_CKPT_PATH \
  --output_path MS_CKPT_PATH \
  --model 'deepseekv3' \
  --num_layers 61 \
  --hidden_size 7168 \
  --ffn_hidden_size 18432 \
  --moe_ffn_hidden_size 2048 \
  --num_routed_experts 256 \
  --num_nextn_predict_layers 1 \
  --first_k_dense_replace 3 \
  --dtype 'bf16'
```

所有指令参数介绍如下：

| 配置项                      | 数据类型   | 是否可选 | 默认值    | 说明                                                                                     |
|--------------------------|--------|------|--------|----------------------------------------------------------------------------------------|
| input_path               | string | 必选   | 无      | 需要转换的 Hugging Face 权重路径。                                                               |
| output_path              | string | 必选   | 无      | 转换后的 MindSpore Transformers 权重目标路径。                                                    |
| model                    | string | 必选   | 无      | 选择进行权重转换的模型，此处对应配置为 `'deepseekv3'`。                                                    |
| num_layers               | int    | 可选   | 61     | 模型层数（计算时不包括 MTP 层数），配置在 Hugging Face 仓库上的 `config.json` 中的 `num_hidden_layers` 。       |
| hidden_size              | int    | 可选   | 7168   | 模型隐藏层大小，配置在 Hugging Face 仓库上的 `config.json` 中的 `hidden_size` 。                         |
| ffn_hidden_size          | int    | 可选   | 18432  | 模型前馈神经网络层的维度，配置在 Hugging Face 仓库上的 `config.json` 中的 `intermediate_size` 。              |
| moe_ffn_hidden_size      | int    | 可选   | 2048   | 模型 MoE 中前馈神经网络层的维度，配置在 Hugging Face 仓库上的 `config.json` 中的 `moe_intermediate_size` 。    |
| num_routed_experts       | int    | 可选   | 256    | 模型专家数，配置在 Hugging Face 仓库上的 `config.json` 中的 `n_routed_experts` 。                      |
| num_nextn_predict_layers | int    | 可选   | 1      | MTP 层数，配置在 Hugging Face 仓库上的 `config.json` 中的 `num_nextn_predict_layers` 。             |
| first_k_dense_replace    | int    | 可选   | 3      | 指定模型的前几层为 Dense 层，配置在 Hugging Face 仓库上的 `config.json` 中的 `first_k_dense_replace` 。     |
| dtype                    | string | 可选   | 'bf16' | 目标转换的 MindSpore Transformers 权重数据类型，可选为 `'bf16'` 、 `'fp16'` 和 `'fp32'` ，默认为 `'bf16'` 。 |

如果转换时不需要 MTP 层（如进行 MindSpore Transformer 推理场景），可以将上述指令的 `--num_nextn_predict_layers` 参数设置为 `0`，例如：

```bash
python convert_weight.py \
  --input_path HF_CKPT_PATH \
  --output_path MS_CKPT_PATH \
  --model 'deepseekv3' \
  --num_nextn_predict_layers 0
```

这样就只会得到包括前 3 层 Dense，和后 58 层 MoE 的权重，不含 MTP 层。

如果转换时只需要前四层，且不需要 MTP 层，可以将上述指令的 `--num_layers` 参数设置为 `4`，并将 `--num_nextn_predict_layers` 参数设置为 `0`：

```bash
python convert_weight.py \
  --input_path HF_CKPT_PATH \
  --output_path MS_CKPT_PATH \
  --model 'deepseekv3' \
  --num_layers 4 \
  --num_nextn_predict_layers 0
```

这样可以使用减层权重体验小规模的微调任务。

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

| 配置项                     | 数据类型   | 是否可选 | 默认值   | 说明                                                                                                                |
|-------------------------|--------|------|-------|-------------------------------------------------------------------------------------------------------------------|
| src_strategy_dirs       | string | 必选   | 无     | 训练时 MindSpore Transformers 的策略文件路径，一般存在 `output/strategy` 文件夹底下。                                                  |
| mindspore_ckpt_dir      | string | 必选   | 无     | 训练时保存的 MindSpore Transformers 训练权重路径，下面存在若干个 "rank_x" 文件夹，存有各卡训练时存下的权重文件。                                         |
| output_dir              | string | 必选   | 无     | 合并后的 MindSpore Transformers 训练权重路径。                                                                               |
| file_suffix             | string | 必选   | 无     | 合并的权重训练步数的前缀，如训练的第 1000 步各卡文件名若为 `deepseekv3_rank_x-1000_1.safetensors`，则此处配置为 `"1000_1"`，表示合并时取文件名中含有该前缀的权重进行合并。 |
| filter_out_param_prefix | string | 必选   | 无     | 去优化器合并时，需要配置此项，且配置为 `"adam_"`。                                                                                    |
| has_redundancy          | bool   | 可选   | True  | 需要合并的权重是否有冗余，若为去冗余保存，则需将此处设置为 `False`。                                                                            |

上述指令权重合并完后，权重会保存在 `'/path/to/unified_train_ckpt'` 文件夹底下的 `'1000_1_ckpt_convert/unified_safe'` 文件夹中。

假设下面使用 `MS_TRAIN_CKPT_PATH` 代指 `'1000_1_ckpt_convert/unified_safe'`，则可以使用反转脚本将权重反转为 Hugging Face 格式， 反转脚本的使用示例指令如下：

```bash
python convert_weight.py \
  --input_path MS_TRAIN_CKPT_PATH \
  --output_path HF_REVERSE_CKPT_PATH \
  --model 'deepseekv3' \
  --reversed \
  --num_layers 61 \
  --hidden_size 7168 \
  --ffn_hidden_size 18432 \
  --moe_ffn_hidden_size 2048 \
  --num_routed_experts 256 \
  --num_nextn_predict_layers 1 \
  --first_k_dense_replace 3 \
  --dtype 'bf16'
```

所有指令参数介绍如下：

| 配置项                      | 数据类型   | 是否可选 | 默认值    | 说明                                                                                                                           |
|--------------------------|--------|------|--------|------------------------------------------------------------------------------------------------------------------------------|
| input_path               | string | 必选   | 无      | 需要转换的 MindSpore Transformers 训练权重路径。                                                                                         |
| output_path              | string | 必选   | 无      | 转换后的 Hugging Face 权重的目标路径。                                                                                                   |
| model                    | string | 必选   | 无      | 选择进行权重转换的模型，此处对应配置为 `'deepseekv3'`。                                                                                          |
| reversed                 | bool   | 必选   | False  | 是否进行权重反转。使用时，仅需设置 `--reversed` 即可，效果与 `--reversed True` 一致。                                                                  |
| num_layers               | int    | 可选   | 61     | 模型层数（计算时不包括 MTP 层数），对应训练 yaml 文件中的 `model.mocel_config.num_layers`（别名可能为`model.mocel_config.num_hidden_layers`）。             |
| hidden_size              | int    | 可选   | 7168   | 模型隐藏层大小，对应训练 yaml 文件中的 `model.mocel_config.hidden_size`。                                                                     |
| ffn_hidden_size          | int    | 可选   | 18432  | 模型前馈神经网络层的维度，对应训练 yaml 文件中的 `model.mocel_config.ffn_hidden_size`。（别名可能为`model.mocel_config.intermediate_size`）。              |
| moe_ffn_hidden_size      | int    | 可选   | 2048   | 模型 MoE 中前馈神经网络层的维度，对应训练 yaml 文件中的 `model.mocel_config.moe_ffn_hidden_size`（别名可能为`model.mocel_config.moe_intermediate_size`）。 |
| num_routed_experts       | int    | 可选   | 256    | 模型专家数，对应训练 yaml 文件中的 `model.mocel_config.num_routed_experts`（别名可能为`model.mocel_config.n_routed_experts`）。                    |
| num_nextn_predict_layers | int    | 可选   | 1      | MTP 层数，对应训练 yaml 文件中的 `model.mocel_config.mtp_num_layers`（别名可能为`model.mocel_config.num_nextn_predict_layers`）。               |
| first_k_dense_replace    | int    | 可选   | 3      | 指定模型的前几层为 Dense 层，对应训练 yaml 文件中的 `model.mocel_config.first_k_dense_replace`。                                                 |
| dtype                    | string | 可选   | 'bf16' | 目标转换的 Hugging Face 权重数据类型，可选为 `'bf16'` 、 `'fp16'` 和 `'fp32'` ，默认为 `'bf16'` 。                                                 |

如果转换时不需要 MTP 层（如权重反转后用于 vLLM 推理场景），可以将上述指令的 `--num_nextn_predict_layers` 参数设置为 `0`，例如：

```bash
python convert_weight.py \
  --input_path MS_TRAIN_CKPT_PATH \
  --output_path HF_REVERSE_CKPT_PATH \
  --model 'deepseekv3' \
  --reversed \
  --num_nextn_predict_layers 0
```

这样就只会得到包括前 3 层 Dense，和后 58 层 MoE 的权重，不含 MTP 层。

如果训练时模型有 1 层 Dense、2 层 MoE 和 1 层 MTP（注意 `num_layers` 的计算不包括 MTP 层，所以为 1 + 2 = 3 层），转换时需要修改 `--num_layers` 、 `--num_nextn_predict_layers` 和 `first_k_dense_replace`：

```bash
python convert_weight.py \
  --input_path MS_TRAIN_CKPT_PATH \
  --output_path HF_REVERSE_CKPT_PATH \
  --model 'deepseekv3' \
  --reversed \
  --num_layers 3 \
  --first_k_dense_replace 1 \
  --num_nextn_predict_layers 1
```
