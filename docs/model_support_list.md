# 模型支持列表

## NLP

### masked_language_modeling

### [text_generation](task_cards/text_generation.md)

|                    模型 <br/> model                     |                                    模型规格 <br/> type                                    |  数据集 <br/> dataset  |           评估指标 <br/> metric           |                           评估得分 <br/> score                            |                                 配置 <br/> config                                  |
|:-----------------------------------------------------:|:-------------------------------------------------------------------------------------:|:-------------------:|:-------------------------------------:|:---------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
|            [llama2](model_cards/llama2.md)            | llama2_7b <br/>llama2_13b <br/> llama2_7b_lora <br/> llama2_13b_lora <br/> llama2_70b |       alpaca        |             PPL / EM / F1             | 6.58 / 39.6 / 60.5 <br/> 6.14 / 27.91 / 44.23 <br/> - <br/> - <br/> - |    [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/llama)     |
|              [glm2](model_cards/glm2.md)              |                              glm2_6b <br/> glm2_6b_lora                               |        ADGEN        | BLEU-4 / Rouge-1 / Rouge-2 / Rouge-l  |     7.47 / 30.78 / 7.07 / 24.77 <br/> 7.23 / 31.06 / 7.18 / 24.23     |     [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/glm2)     |
|              [glm3](model_cards/glm3.md)              |                                        glm3_6b                                        |        ADGEN        |                   -                   |                                   -                                   |     [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/glm3)     |
|              [gpt2](model_cards/gpt2.md)              |                               gpt2_small <br/> gpt2_13b                               |     wikitext-2      |                   -                   |                                   -                                   |     [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/gpt2)     |
|    [baichuan2](../research/baichuan2/baichuan2.md)    |  baichuan2_7b <br/> baichuan2_13b  <br/> baichuan2_7b_lora <br/> baichuan2_13b_lora   |        belle        |                   -                   |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/baichuan2)  |
|         [codegeex2](model_cards/codegeex2.md)         |                                     codegeex2_6b                                      |     CodeAlpaca      |                   -                   |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/codegeex2)   |
|         [codellama](model_cards/codellama.md)         |                                     codellama_34b                                     |     CodeAlpaca      |                   -                   |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/dev/configs/codellama)   |
|     [deepseek](../research/deepseek/deepseek.md)      |                                     deepseek_33b                                      |          -          |                   -                   |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/deepseek)   |
|     [internlm](../research/internlm/internlm.md)      |                            internlm_7b <br/> internlm_20b                             |       alpaca        |                   -                   |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/internlm)   |
|       [mixtral](../research/mixtral/mixtral.md)       |                                     mixtral-8x7b                                      |     wikitext-2      |                   -                   |                                   -                                   |   [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/mixtral)   |
|           [qwen](../research/qwen/qwen.md)            |             qwen_7b <br/> qwen_14b <br/> qwen_7b_lora <br/> qwen_14b_lora             |       alpaca        |                C-Eval                 |                   63.3 <br/> 72.13 <br/> - <br/> -                    |    [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/qwen)     |
|       [qwen1.5](../research/qwen1_5/qwen1_5.md)       |                             qwen1.5-14b <br/> qwen1.5-72b                             |          -          |                   -                   |                                   -                                   |   [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/qwen1_5)   |
| [wizardcoder](../research/wizardcoder/wizardcoder.md) |                                    wizardcoder_15b                                    |     CodeAlpaca      |              MBPP Pass@1              |                                 50.8                                  | [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/wizardcoder) |
|              [yi](../research/yi/yi.md)               |                                  yi_6b <br/> yi_34b                                   | alpaca_gpt4_data_zh |                   -                   |                                   -                                   |     [configs](https://gitee.com/mindspore/mindformers/tree/dev/research/yi)      |

## LLM大模型能力支持一览

|     模型  \  特性     |      低参微调       |      边训边评      | Flash Attention | 并行推理  | 流式推理 | Chat | 多轮对话 |
|:-----------------:|:---------------:|:--------------:|:---------------:|:-----:|:----:|:----:|:----:|
| Llama2-7B/13B/70B |      Lora       |      PPL       |        √        | dp/mp |  √   |  √   |  √   |
|      GLM2-6B      | Lora/P-TuningV2 | PPL/Bleu/Rouge |        √        | dp/mp |  √   |  √   |  √   |
|      GLM3-6B      |      Lora       |       ×        |        √        | dp/mp |  √   |  √   |  √   |
|   CodeGeex2-6B    |        ×        | PPL/Bleu/Rouge |        √        | dp/mp |  √   |  √   |  √   |
|   CodeLlama-34B   |      Lora       |     pass@1     |        √        | dp/mp |  √   |  √   |  ×   |
|   GPT2-128m/13B   |      Lora       |      PPL       |        √        | dp/mp |  √   |  ×   |  ×   |
| BaiChuan2-7B/13B  |      Lora       |      PPL       |        √        | dp/mp |  √   |  √   |  √   |
|    Qwen-7B/14B    |        √        |       ×        |        √        | dp/mp |  √   |  √   |  √   |
|    Qwen1.5-14B    |        ×        |       ×        |        ×        | dp/mp |  √   |  √   |  √   |
|    Qwen1.5-72B    |      Lora       |       ×        |        √        | dp/mp |  √   |  √   |  √   |
|  InternLM-7B/20B  |      Lora       |      PPL       |        √        | dp/mp |  √   |  √   |  √   |
|  Wizardcoder-15B  |        ×        |      PPL       |        ×        | dp/mp |  √   |  √   |  √   |
|   Deepseek-33B    |        ×        |       ×        |        ×        |   ×   |  ×   |  √   |  ×   |
|   Mixtral-8×7B    |        ×        |       ×        |        √        |   √   |  ×   |  ×   |  ×   |
|       Yi-6B       |      Lora       |       ×        |        √        | dp/mp |  √   |  √   |  √   |
|      Yi-34B       |        ×        |       ×        |        ×        | dp/mp |  √   |  ×   |  ×   |
