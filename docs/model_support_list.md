# 模型支持列表

## NLP

### masked_language_modeling

|             模型 <br> model              | 数据集 <br> dataset | 评估指标 <br> metric | 评估得分 <br> score |                                                     配置<br>config                                                     |
| :--------------------------------------: | :-----------------: | :------------------: | :-----------------: | :--------------------------------------------------------------------------------------------------------------------: |
| [bert_base_uncased](model_cards/bert.md) |        wiki         |          -           |          -          | [run_bert_base_uncased.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/bert/run_bert_base_uncased.yaml) |

### [text_classification](task_cards/text_classification.md)

|                                                            模型 <br> model                                                            | 数据集 <br> dataset |   评估指标 <br> metric   | 评估得分 <br> score |                                                                                                                                        配置<br>config                                                                                                                                        |
| :-----------------------------------------------------------------------------------------------------------------------------------: | :-----------------: | :----------------------: | :-----------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [txtcls_bert_base_uncased](task_cards/text_classification.md) <br> [txtcls_bert_base_uncased_mnli](task_cards/text_classification.md) |   Mnli <br> Mnli    | Entity F1 <br> Entity F1 |   - <br>   84.80%   | [run_txtcls_bert_base_uncased.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/txtcls/run_txtcls_bert_base_uncased.yaml) <br> [run_txtcls_bert_base_uncased_mnli.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/txtcls/run_txtcls_bert_base_uncased_mnli.yaml) |

### [token_classification](task_cards/token_classification.md)

|                                                              模型 <br> model                                                               | 数据集 <br> dataset  |   评估指标 <br> metric   | 评估得分 <br> score |                                                                                                                                           配置<br>config                                                                                                                                           |
| :----------------------------------------------------------------------------------------------------------------------------------------: | :------------------: | :----------------------: | :-----------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [tokcls_bert_base_chinese](task_cards/token_classification.md) <br> [tokcls_bert_base_chinese_cluener](task_cards/token_classification.md) | CLUENER <br> CLUENER | Entity F1 <br> Entity F1 |  - <br>    0.7905   | [run_tokcls_bert_base_chinese.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/tokcls/run_tokcls_bert_base_chinese.yaml) <br> [run_tokcls_bert_base_chinese_cluener.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/tokcls/run_tokcls_bert_base_chinese_cluener.yaml) |

### [question_answering](task_cards/question_answering.md)

|                                                        模型 <br> model                                                         |    数据集 <br> dataset     | 评估指标 <br> metric | 评估得分 <br> score  |                                                                                                                               配置<br>config                                                                                                                               |
| :----------------------------------------------------------------------------------------------------------------------------: | :------------------------: | :------------------: | :------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [qa_bert_base_uncased](task_cards/question_answering.md) <br> [qa_bert_base_chinese_uncased](task_cards/question_answering.md) | SQuAD v1.1 <br> SQuAD v1.1 | EM / F1 <br> EM / F1 | 80.74 / 88.33 <br> - | [run_qa_bert_base_uncased.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/qa/run_qa_bert_base_uncased.yaml) <br> [run_qa_bert_base_chinese_uncased.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/qa/run_qa_bert_base_chinese_uncased.yaml) |

### translation

|        模型 <br> model        | 数据集 <br> dataset | 评估指标 <br> metric | 评估得分 <br> score |                                                    配置<br>config                                                    |
| :---------------------------: | :-----------------: | :------------------: | :-----------------: | :------------------------------------------------------------------------------------------------------------------: |
| [t5_small](model_cards/t5.md) |        WMT16        |          -           |          -          | [run_t5_small_on_wmt16.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/t5/run_t5_small_on_wmt16.yaml) |

### [text_generation](task_cards/text_generation.md)

|                                                                    模型 <br> model                                                                     |              数据集 <br> dataset               |             评估指标 <br> metric             |                     评估得分 <br> score                      |                                                                                                                                                                                                                    配置<br>config                                                                                                                                                                                                                    |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------: | :------------------------------------------: | :----------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  [bloom_560m](model_cards/bloom.md)<br>[bloom_7.1b](model_cards/bloom.md) <br>[bloom_65b](model_cards/bloom.md)<br>[bloom_176b](model_cards/bloom.md)  | alpaca <br> alpaca <br> alpaca <br>     alpaca |         - <br>   - <br>   - <br>   -         |                 - <br>   - <br>   - <br>   -                 | [run_bloom_560m.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/bloom/run_bloom_560m.yaml) <br> [run_bloom_7.1b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/bloom/run_bloom_7.1b.yaml) <br> [run_bloom_65b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/bloom/run_bloom_65b.yaml) <br> [run_bloom_176b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/bloom/run_bloom_176b.yaml) |
|                                           [glm_6b](model_cards/glm.md)<br>[glm_6b_lora](model_cards/glm.md)                                            |                ADGEN <br> ADGEN                | BLEU-4 / Rouge-1 / Rouge-2 / Rouge-l <br>  - |              8.42 / 31.75 / 7.98 / 25.28 <br> -              |                                                                                                           [run_glm_6b_finetune.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/glm/run_glm_6b_finetune.yaml) <br> [run_glm_6b_lora.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/glm/run_glm_6b_lora.yaml)                                                                                                           |
|                                         [glm2_6b](model_cards/glm2.md)<br>[glm2_6b_lora](model_cards/glm2.md)                                          |                ADGEN <br> ADGEN                | BLEU-4 / Rouge-1 / Rouge-2 / Rouge-l <br>  - | 7.47 / 30.78 / 7.07 / 24.77 <br> 7.23 / 31.06 / 7.18 / 24.23 |                                                                                                                 [run_glm2_6b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/glm2/run_glm2_6b.yaml) <br> [run_glm2_6b_lora.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/glm2/run_glm2_6b_lora.yaml)                                                                                                                 |
|                       [gpt2_small](model_cards/gpt2.md) <br>[gpt2_13b](model_cards/gpt2.md) <br>[gpt2_52b](model_cards/gpt2.md)                        |   wikitext-2 <br> wikitext-2 <br> wikitext-2   |             - <br>   - <br>   -              |                     - <br>   - <br>   -                      |                                                                  [run_gpt2.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2.yaml) <br>  [run_gpt2_13b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2_13b.yaml) <br>  [run_gpt2_52b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2_52b.yaml)                                                                  |
| [llama_7b](model_cards/llama.md) <br>[llama_13b](model_cards/llama.md) <br>[llama_65b](model_cards/llama.md) <br>[llama_7b_lora](model_cards/llama.md) |     alpaca <br> alpaca <br> alpaca <br> -      |         - <br>   - <br>   - <br>   -         |                 - <br>   - <br>   - <br>   -                 | [run_llama_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama/run_llama_7b.yaml) <br> [run_llama_13b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama/run_llama_13b.yaml) <br> [run_llama_65b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama/run_llama_65b.yaml) <br> [run_llama_7b_lora.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama/run_llama_7b_lora.yaml) |
| [llama_7b](model_cards/llama.md) <br>[llama_13b](model_cards/llama.md) <br>[llama_65b](model_cards/llama.md) <br>[llama_7b_lora](model_cards/llama.md) |     alpaca <br> alpaca <br> alpaca <br> -      |         - <br>   - <br>   - <br>   -         |                 - <br>   - <br>   - <br>   -                 | [run_llama_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama/run_llama_7b.yaml) <br> [run_llama_13b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama/run_llama_13b.yaml) <br> [run_llama_65b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama/run_llama_65b.yaml) <br> [run_llama_7b_lora.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama/run_llama_7b_lora.yaml) |
|                              [pangualpha_2_6_b](model_cards/pangualpha.md)<br>[pangualpha_13b](model_cards/pangualpha.md)                              |           悟道数据集 <br> 悟道数据集           |                  - <br>   -                  |                          - <br>   -                          |                                                                                                 [run_pangualpha_2_6b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/pangualpha/run_pangualpha_2_6b.yaml) <br> [run_pangualpha_13b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/pangualpha/run_pangualpha_13b.yaml)                                                                                                 |
|                          [baichuan_7b](../research/baichuan/baichuan.md)<br>[baichuan_13b](../research/baichuan/baichuan.md)                           |                    - <br> -                    |                  - <br>  -                   |                           - <br> -                           |                                                                                                        [run_baichuan_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/baichuan/run_baichuan_7b.yaml) <br> [run_baichuan_13b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/baichuan/run_baichuan_13b.yaml)                                                                                                        |
|                       [baichuan2_7b](../research/baichuan2/baichuan2.md)<br>[baichuan2_13b](../research/baichuan2/baichuan2.md)                        |                    - <br> -                    |                  - <br>  -                   |                           - <br> -                           |                                                                                                     [run_baichuan2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/baichuan2/run_baichuan2_7b.yaml) <br> [run_baichuan2_13b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/baichuan2/run_baichuan2_13b.yaml)                                                                                                     |
|                        [internlm_7b](../research/internlm/internlm.md)<br>[internlm_7b_lora](../research/internlm/internlm.md)                         |             wikitext-2 <br> alpaca             |                  - <br>  -                   |                           - <br> -                           |                                                                                                    [run_internlm_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/internlm/run_internlm_7b.yaml) <br> [run_internlm_7b_lora.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/internlm/run_internlm_7b_lora.yaml)                                                                                                    |
|                                                          [ziya_13b](../research/ziya/ziya.md)                                                          |                    - <br> -                    |                  - <br>  -                   |                           - <br> -                           |                                                                                                                                                                      [run_ziya_13b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/baichuan/run_ziya_13b.yaml)                                                                                                                                                                       |

## CV

### masked_image_modeling

|            模型 <br> model             | 数据集 <br> dataset | 评估指标 <br> metric | 评估得分 <br> score |                                                             配置<br>config                                                              |
| :------------------------------------: | :-----------------: | :------------------: | :-----------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| [mae_vit_base_p16](model_cards/mae.md) |     ImageNet-1k     |          -           |          -          | [run_mae_vit_base_p16_224_800ep.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/mae/run_mae_vit_base_p16_224_800ep.yaml) |

### [image_classification](task_cards/image_classification.md)

|            模型 <br> model            | 数据集 <br> dataset | 评估指标 <br> metric | 评估得分 <br> score |                                                            配置<br>config                                                            |
| :-----------------------------------: | :-----------------: | :------------------: | :-----------------: | :----------------------------------------------------------------------------------------------------------------------------------: |
|  [vit_base_p16](model_cards/vit.md)   |     ImageNet-1k     |       Accuracy       |       83.71%        |   [run_vit_base_p16_224_100ep.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/vit/run_vit_base_p16_224_100ep.yaml)    |
| [swin_base_p4w7](model_cards/swin.md) |     ImageNet-1k     |       Accuracy       |       83.44%        | [run_swin_base_p4w7_224_100ep.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/swin/run_swin_base_p4w7_224_100ep.yaml) |

## Multi-Modal

### [zero_shot_image_classification](task_cards/zero_shot_image_classification.md) (by [contrastive_language_image_pretrain](task_cards/contrastive_language_image_pretrain.md))

|                                                                          模型 <br> model                                                                          |                数据集 <br> dataset                 |                    评估指标 <br> metric                     |            评估得分 <br> score             |                                                                                                                                                                                                                                                                                                           配置<br>config                                                                                                                                                                                                                                                                                                           |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------: | :---------------------------------------------------------: | :----------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [clip_vit_b_32](model_cards/clip.md)<br>[clip_vit_b_16](model_cards/clip.md) <br>[clip_vit_l_14](model_cards/clip.md)<br>[clip_vit_l_14@336](model_cards/clip.md) | Cifar100 <br> Cifar100 <br> Cifar100 <br> Cifar100 | Accuracy   <br>  Accuracy   <br>  Accuracy   <br>  Accuracy | 57.24% <br> 61.41% <br> 69.67% <br> 68.19% | [run_clip_vit_b_32_pretrain_flickr8k.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/clip/run_clip_vit_b_32_pretrain_flickr8k.yaml) <br> [run_clip_vit_b_16_pretrain_flickr8k.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/clip/run_clip_vit_b_16_pretrain_flickr8k.yaml) <br> [run_clip_vit_l_14_pretrain_flickr8k.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/clip/run_clip_vit_l_14_pretrain_flickr8k.yaml) <br> [run_clip_vit_l_14@336_pretrain_flickr8k.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/clip/run_clip_vit_l_14@336_pretrain_flickr8k.yaml) |
|                                                                [blip2_vit_g](model_cards/blip2.md)                                                                |              - <br> flickr30k <br> -               |                      - <br> ITM <br> -                      |              - <br> - <br> -               |                                                              [run_blip2_vit_g_qformer_pretrain.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/blip2/run_blip2_vit_g_qformer_pretrain.yaml) <br> [run_blip2_vit_g_retrieval_flickr30k.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/blip2/run_blip2_vit_g_retrieval_flickr30k.yaml) <br> [run_blip2_vit_g_zero_shot_image_classification_cifar100.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/blip2/run_blip2_vit_g_zero_shot_image_classification_cifar100.yaml)                                                               |

## 模型能力支持度

### 核心关键模型能力一览表

| 关键模型 |             并行模式             | 数据并行 | 优化器并行 | 模型并行 | 流水并行 | 多副本并行 | 预训练 |        微调        |      评估      | 推理 |
| :------: | :------------------------------: | :------: | :--------: | :------: | :------: | :--------: | ------ | :----------------: | :------------: | ---: |
|  Bloom   | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     |      全参微调      |     不支持     | 推理 |
|   GLM    | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     | 全参微调，Lora微调 | Blue/Rouge评估 | 推理 |
|   GLM2   | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     | 全参微调，Lora微调 | Blue/Rouge评估 | 推理 |
|   GPT    | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     |      全参微调      |    PPL评估     | 推理 |
|  LLaMa   | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     | 全参微调，Lora微调 |    PPL评估     | 推理 |
|  LLaMa2  | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     |      全参微调      |    PPL评估     | 推理 |
|  PanGu   | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     |      全参微调      |    PPL评估     | 推理 |

### Research模型支持情况一览表

|                      模型                      |                                           任务（task name）                                            | 模型（model name）              |
| :--------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :------------------------------ |
|  [Baichuan](../research/baichuan/baichuan.md)  | [text_generation](https://gitee.com/mindspore/mindformers/blob/dev/docs/task_cards/text_generation.md) | baichuan_7b <br> baichuan_13b   |
| [Baichuan2](../research/baichuan2/baichuan2.md) | [text_generation](https://gitee.com/mindspore/mindformers/blob/dev/docs/task_cards/text_generation.md) | baichuan2_7b <br> baichuan2_13b |
|  [Internlm](../research/internlm/internlm.md)  | [text_generation](https://gitee.com/mindspore/mindformers/blob/dev/docs/task_cards/text_generation.md) | InternLM-7B                     |
|        [ziya](../research/ziya/ziya.md)        | [text_generation](https://gitee.com/mindspore/mindformers/blob/dev/docs/task_cards/text_generation.md) | ziya-13B                        |

### Text Generator支持度表

|    model    |                 模型文档链接                  | 增量推理 | 流式推理 |
| :---------: | :-------------------------------------------: | :------: | :------: |
|    bloom    |        [link](../model_cards/bloom.md)        |    √     |    √     |
|     GLM     |         [link](../model_cards/glm.md)         |    √     |    √     |
|    GLM2     |        [link](../model_cards/glm2.md)         |    √     |    √     |
|     GPT     |        [link](../model_cards/gpt2.md)         |    √     |    √     |
|    llama    |        [link](../model_cards/llama.md)        |    √     |    √     |
|   llama2    |        [link](../model_cards/llama.md)        |    √     |    √     |
| pangu-alpha |     [link](../model_cards/pangualpha.md)      |    √     |    √     |
|     T5      |         [link](../model_cards/t5.md)          |    ×     |    √     |
|  research   |                   research                    | research | research |
|  baichuan   |  [link](../../research/baichuan/baichuan.md)  |    √     |    √     |
|  baichuan2  | [link](../../research/baichuan2/baichuan2.md) |    √     |    √     |
|  internlm   |  [link](../../research/internlm/internlm.md)  |    √     |    √     |
|    ziya     |      [link](../../research/ziya/ziya.md)      |    √     |    √     |

### 边训练边评估支持度表

| 模型        |       评估指标        | 可用Model.eval完成评估 | 是否支持 | 数据并行模式 |  半自动并行模式   |
| ----------- | :-------------------: | :--------------------: | -------- | ------------ | :---------------: |
| bert        |           -           |           -            | -        | -            |         -         |
| blip2       |           -           |           -            | -        | -            |         -         |
| bloom       |           -           |           -            | -        | -            |         -         |
| clip        |           -           |           -            | -        | -            |         -         |
| filip       |           -           |           -            | -        | -            |         -         |
| glm         |      Rouge，Bleu      |           否           | 否       | ×            |         ×         |
| gpt2        |          PPL          |           是           | 是       | √            |         √         |
| llama       |          PPL          |           是           | 是       | √            |  √（7b 至少8卡）  |
| llama2      |          PPL          |           是           | 是       | √            |  √（7b 至少8卡）  |
| MAE         |         暂缺          |           -            | -        | -            |         -         |
| pangu alpha |          PPL          |           是           | 是       | √            |         √         |
| qa-bert     | f1, precision, recall |           是           | 是       | √            |         ×         |
| swin        |       Accuracy        |           是           | 是       | √            |         ×         |
| t5          |         暂缺          |           -            | -        | -            |         -         |
| tokcls-bert | f1, precision, recall |           是           | 是       | √            |         ×         |
| txtcls-bert |       Accuracy        |           是           | 是       | √            |         ×         |
| vit         |       Accuracy        |           是           | 是       | √            |         ×         |
| research    |       research        |        research        | research | research     |     research      |
| baichuan    |          PPL          |           是           | 是       | √            |  √（7b 至少8卡）  |
| baichuan2   |          PPL          |           是           | 是       | √            |  √（7b 至少8卡）  |
| internlm    |          PPL          |           是           | 是       | √            |  √（7b 至少8卡）  |
| ziya        |          PPL          |           是           | 是       | √            | √（13b 至少16卡） |

### 微调支持列表

|               模型               | 微调算法 |        运行模式         |
| :------------------------------: | :------: | :---------------------: |
|  [GPT2](../model_cards/gpt2.md)  |   Lora   | finetune、eval、predict |
| [LLama](../model_cards/llama.md) |   Lora   | finetune、eval、predict |
|   [GLM](../model_cards/glm.md)   |   Lora   | finetune、eval、predict |
|  [GLM2](../model_cards/glm2.md)  |   Lora   | finetune、eval、predict |

### Chat Web支持列表

| 模型  | 规格          | 分词器        | 增量推理 |
| ----- | ------------- | ------------- | -------- |
| GLM   | glm_6b        | glm_6b        | 支持     |
| GLM2  | glm2_6b       | glm2_6b       | 支持     |
| BLOOM | bloom_7.1b    | bloom_7.1b    | 支持     |
| LLAMA | llama_7b_lora | llama_7b_lora | 支持     |

### 其余库上模型分布式支持情况一览表

| 模型  | 并行模式      | 数据并行 | 优化器并行 | 模型并行 | 流水并行 | 多副本并行 |
| ----- | ------------- | -------- | ---------- | -------- | -------- | ---------- |
| Bert  | data_parallel | 是       | 是         | 否       | 否       | 否         |
| BLIP2 | data_parallel | 是       | 是         | 否       | 否       | 否         |
| CLIP  | data_parallel | 是       | 是         | 否       | 否       | 否         |
| MAE   | data_parallel | 是       | 是         | 否       | 否       | 否         |
| Swin  | data_parallel | 是       | 是         | 否       | 否       | 否         |
| T5    | data_parallel | 是       | 是         | 否       | 否       | 否         |
| VIT   | data_parallel | 是       | 是         | 否       | 否       | 否         |
