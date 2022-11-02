# Benchmark

## BERT

基于HuggingFace中BERT-large的[checkpoint](https://huggingface.co/bert-large-uncased/tree/main)情况下进行微调，
下表给出了在GLUE上的微调精度结果。

| Task        | MNLI | QNLI   | SST-2 | MRPC | RTE |  Squad|
|-------------|------|--------|-------|------| ------|------|
| HuggingFace | 86.1 | 92.3  |    93.2   |   88.0   |  70.1 |  90.7|
| Ours        | 86.3 | 92.4 |     93.6  |  88.6  | 70.2 | 90.8 |

### 环境

| 项目   | 值   |
|------| --------- |
| 模型规模 | bert-base |
| 环境   | A100     |
 | MindSpore | 2.0.0 |

## GPT

基于HuggingFace中GPT2的[checkpoint](https://huggingface.co/gpt2/tree/main)情况下微调，评测PPL指标，
对应结果如下。

| Task        | Wikitext-2 |
|-------------|------------|
| HuggingFace | 17.8  |
| Ours        | 17.7       |

### 环境

| 项目   | 值     |
|------|-------|
| 模型规模 | gpt2  |
| 环境   | A100  |
| MindSpore | 2.0.0 |

