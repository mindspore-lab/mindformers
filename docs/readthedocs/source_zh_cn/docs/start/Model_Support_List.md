# 模型特性支持列表

**此处给出了MindFormers套件中支持的任务名称和模型名称，用于高阶开发时的索引名**

|                             模型                             |                      任务（task name）                       | 模型（model name）                                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------- |
| [BERT](../model_cards/bert.md) | masked_language_modeling<br>[text_classification](../task_cards/text_classification.md)<br>[token_classification](../task_cards/token_classification.md)<br>[question_answering](../task_cards/question_answering.md) | bert_base_uncased <br>txtcls_bert_base_uncased<br>txtcls_bert_base_uncased_mnli <br>tokcls_bert_base_chinese<br>tokcls_bert_base_chinese_cluener <br>qa_bert_base_uncased<br>qa_bert_base_chinese_uncased |
| [T5](../model_cards/t5.md) |                         translation                          | t5_small                                                     |
| [GPT2](../model_cards/gpt2.md) |                       text_generation                        | gpt2_small <br>gpt2_13b <br>gpt2_52b                         |
| [PanGuAlpha](../model_cards/pangualpha.md) |                       text_generation                        | pangualpha_2_6_b<br>pangualpha_13b                           |
| [GLM](../model_cards/glm.md) |                       text_generation                        | glm_6b<br>glm_6b_lora                                        |
| [LLama](../model_cards/llama.md) |                       text_generation                        | llama_7b <br>llama_13b <br>llama_65b <br>llama_7b_lora       |
|                            [Bloom](../model_cards/bloom.md)                             |                       text_generation                        | bloom_560m<br>bloom_7.1b <br>bloom_65b<br>bloom_176b         |
| [MAE](../model_cards/mae.md) |                    masked_image_modeling                     | mae_vit_base_p16                                             |
| [VIT](../model_cards/vit.md) | [image_classification](../task_cards/image_classification.md) | vit_base_p16                                                 |
| [Swin](../model_cards/swin.md) | [image_classification](../task_cards/image_classification.md) | swin_base_p4w7                                               |
| [CLIP](../model_cards/clip.md) | [contrastive_language_image_pretrain](../task_cards/contrastive_language_image_pretrain.md)<br>[zero_shot_image_classification](../task_cards/zero_shot_image_classification.md) | clip_vit_b_32<br>clip_vit_b_16 <br>clip_vit_l_14<br>clip_vit_l_14@336 |

**核心关键模型能力一览表：**

| 关键模型 |             并行模式             | 数据并行 | 优化器并行 | 模型并行 | 流水并行 | 多副本并行 | 预训练 |        微调        |      评估      | 推理 |
| :------: | :------------------------------: | :------: | :--------: | :------: | :------: | :--------: | ------ | :----------------: | :------------: | ---: |
|   GPT    | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     |      全参微调      |    PPL评估     | 推理 |
|  PanGu   | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     |      全参微调      |    PPL评估     | 推理 |
|  Bloom   | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     |      全参微调      |     不支持     | 推理 |
|  LLaMa   | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     | 全参微调，Lora微调 |    PPL评估     | 推理 |
|   GLM    | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     | 全参微调，Lora微调 | Blue/Rouge评估 | 推理 |

**其余库上模型分布式支持情况一览表：**

| 模型 | 并行模式      | 数据并行 | 优化器并行 | 模型并行 | 流水并行 | 多副本并行 |
| ---- | ------------- | -------- | ---------- | -------- | -------- | ---------- |
| MAE  | data_parallel | 是       | 是         | 否       | 否       | 否         |
| T5   | data_parallel | 是       | 是         | 否       | 否       | 否         |
| Bert | data_parallel | 是       | 是         | 否       | 否       | 否         |
| Swin | data_parallel | 是       | 是         | 否       | 否       | 否         |
| VIT  | data_parallel | 是       | 是         | 否       | 否       | 否         |
| CLIP | data_parallel | 是       | 是         | 否       | 否       | 否         |

**模型评测支持情况一览表：**

| 模型        |       评估指标        | 可用Model.eval完成评估 | 是否支持 | 数据并行模式 | 半自动并行模式  |
| ----------- | :-------------------: | :--------------------: | -------- | ------------ | :-------------: |
| bert        |           -           |           -            | -        | -            |        -        |
| bloom       |           -           |           -            | -        | -            |        -        |
| clip        |           -           |           -            | -        | -            |        -        |
| filip       |           -           |           -            | -        | -            |        -        |
| glm         |      Rouge，Bleu      |           否           | 否       | ×            |        ×        |
| gpt2        |          PPL          |           是           | 是       | √            |        √        |
| llama       |          PPL          |           是           | 是       | √            | √（7b 至少8卡） |
| MAE         |         暂缺          |           -            | -        | -            |        -        |
| pangu alpha |          PPL          |           是           | 是       | √            |        √        |
| qa-bert     | f1, precision, recall |           是           | 是       | √            |        ×        |
| swin        |       Accuracy        |           是           | 是       | √            |        ×        |
| t5          |         暂缺          |           -            | -        | -            |        -        |
| tokcls-bert | f1, precision, recall |           是           | 是       | √            |        ×        |
| txtcls-bert |       Accuracy        |           是           | 是       | √            |        ×        |
| vit         |       Accuracy        |           是           | 是       | √            |        ×        |

**Text Generator支持情况一览表：**

|    model    |                         模型文档链接                         | 增量推理 | 流式推理 |
| :---------: | :----------------------------------------------------------: | :------: | :------: |
|    bloom    | [link](../model_cards/bloom.md) |    √     |    √     |
|     GLM     | [link](../model_cards/glm.md) |    √     |    √     |
|     GPT     | [link](../model_cards/gpt2.md) |    ×     |    √     |
|    llama    | [link](../model_cards/llama.md) |    √     |    √     |
| pangu-alpha | [link](../model_cards/pangualpha.md) |    ×     |    √     |
|     T5      | [link](../model_cards/t5.md) |    ×     |    √     |
