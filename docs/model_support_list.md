# 模型支持列表

> ## 🚨 弃用说明
>
> 本文档已过时，不再进行维护，并将在 *1.6.0* 版本下架，其中可能包含过时的信息或已被更新的功能替代。建议参考最新的 **[官方文档](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/index.html)** ，以获取准确的信息。
>
> 如果您仍需使用本文档中的内容，请仔细核对其适用性，并结合最新版本的相关资源进行验证。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## NLP

### [text_generation](task_cards/text_generation.md)

|                      模型 <br> model                      |                                     模型规格<br/>type                                      | 数据集 <br> dataset |               评估指标 <br> metric               |                            评估得分 <br> score                            |                                  配置<br>config                                  |
|:-------------------------------------------------------:|:--------------------------------------------------------------------------------------:|:----------------:|:--------------------------------------------:|:---------------------------------------------------------------------:|:------------------------------------------------------------------------------:|
|             [llama2](model_cards/llama2.md)             | llama2_7b <br/> llama2_13b <br/> llama2_7b_lora <br/> llama2_13b_lora <br/> llama2_70b |      alpaca      |                PPL / EM / F1                 | 6.58 / 39.6 / 60.5 <br/> 6.14 / 27.91 / 44.23 <br/> - <br/> - <br/> - |   [configs](https://gitee.com/mindspore/mindformers/tree/r1.5.0/configs/llama2)   |
|         [llama3](../research/llama3/llama3.md)          |                               llama3_8b <br/> llama3_70b                               |      alpaca      |                      -                       |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/llama3)   |
|               [glm3](model_cards/glm3.md)               |                                        glm3_6b                                         |      ADGEN       |                      -                       |                                   -                                   |    [configs](https://gitee.com/mindspore/mindformers/tree/r1.5.0/configs/glm3)    |
|          [codellama](model_cards/codellama.md)          |                                     codellama_34b                                      |    CodeAlpaca    |                      -                       |                                   -                                   | [configs](https://gitee.com/mindspore/mindformers/tree/r1.5.0/configs/codellama)  |
|   [deepseek coder](../research/deepseek/deepseek.md)    |                                      deepseek_33b                                      |    CodeAlpaca    |                      -                       |                                   -                                   | [configs](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/deepseek)  |
|         [glm32k](../research/glm32k/README.md)          |                                      glm3_6b_32k                                       |    LongBench     |                      -                       |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/glm32k)   |
|        [Qwen1.5](../research/qwen1_5/qwen1_5.md)        |                     qwen1_5_7b <br/> qwen1_5_14b <br/> qwen1_5_72b                     |      alpaca      |                      -                       |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/qwen1_5)  |
|     [internlm2](../research/internlm2/internlm2.md)     |                            internlm2_7b <br/> internlm2_20b                            |      alpaca      |                      -                       |                                   -                                   | [configs](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/internlm2) |
|        [mixtral](../research/mixtral/mixtral.md)        |                                      mixtral_8x7b                                      |    wikitext-2    |                      -                       |                                   -                                   |  [configs](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/mixtral)  |

## Multi-Modal

### image_to_text_generation

|             模型 <br> model              | 模型规格<br/>type    |   数据集 <br> dataset    | 评估指标 <br> metric | 评估得分 <br> score |                                                      配置<br>config                                                      |
|:--------------------------------------:|------------------|:---------------------:|:----------------:|:---------------:|:----------------------------------------------------------------------------------------------------------------------:|
| [QwenVL](../research/qwenvl/README.md) | qwenvl_9.6b_bf16 | LLaVa-150k detail_23k |        -         |        -        | [configs](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/qwenvl/qwenvl_9.6b/finetune_qwenvl_9.6b_bf16.yaml) |

## LLM大模型能力支持一览

|     模型  \  特性     | 低参微调 |      边训边评      | Flash Attention | 并行推理  |  流式推理   |  Chat   |  多轮对话   |
|:-----------------:|:----:|:--------------:|:---------------:|:-----:|:-------:|:-------:|:-------:|
| Llama2-7B/13B/70B | Lora |      PPL       |     &check;     | dp/mp | &check; | &check; | &check; |
|   Llama3-8B/70B   |  -   |       -        |     &check;     | dp/mp | &check; | &check; | &check; |
|   CodeLlama-34B   | Lora |   HumanEval    |     &check;     | dp/mp | &check; |    -    |    -    |
|      GLM2-6B      | Lora | PPL/Bleu/Rouge |     &check;     | dp/mp | &check; | &check; | &check; |
|      GLM3-6B      |  -   |       -        |     &check;     | dp/mp | &check; | &check; | &check; |
|    GLM3-6B-32k    |  -   |       -        |     &check;     | dp/mp | &check; | &check; | &check; |
|   GPT2-128m/13B   | Lora |      PPL       |     &check;     | dp/mp | &check; |    -    |    -    |
| BaiChuan2-7B/13B  | Lora |      PPL       |     &check;     | dp/mp | &check; | &check; | &check; |
|    Qwen-7B/14B    | Lora |       -        |     &check;     | dp/mp | &check; | &check; | &check; |
|    QwenVL-9.6B    |  -   |       -        |     &check;     | dp/mp | &check; |    -    |    -    |
|  Qwen-7B/14B/72B  |  -   |       -        |     &check;     | dp/mp | &check; | &check; | &check; |
|  InternLM-7B/20B  | Lora |      PPL       |     &check;     | dp/mp | &check; | &check; | &check; |
| InternLM2-7B/20B  |  -   |       -        |     &check;     | dp/mp | &check; | &check; | &check; |
|     Yi-6B/34B     | Lora |       -        |     &check;     | dp/mp | &check; | &check; | &check; |
|   Mixtral-8x7B    | Lora |       -        |     &check;     | dp/mp | &check; |    -    |    -    |
|   DeepSeek-33B    | Lora |       -        |     &check;     | dp/mp | &check; |    -    |    -    |
