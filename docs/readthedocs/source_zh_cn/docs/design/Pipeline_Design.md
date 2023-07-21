# Pipeline

MindFormers大模型套件面向任务设计pipeline推理接口，旨在让用户可以便捷的体验不同AI领域的大模型在线推理服务，当前已集成10+任务的推理流程；

![输入图片说明](https://foruda.gitee.com/images/1673432339378334189/fb24c2fe_9324149.png "image-20230104093648200.png")

MindFormers大模型套件为用户提供了pipeline高阶API，支持用户便捷的使用套件中已经集成的任务和模型完成推理流程。

**MindFormers 任务推理支持情况一览表：**

|                             任务                             | 支持模型                                                     | 支持推理数据     |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ---------------- |
| [text_generation](../task_cards/text_generation.md) | gpt2<br/>gpt2_13b<br/>gpt2_52b<br/>pangualpha_2_6_b<br/>pangualpha_13b<br/>glm_6b<br/>glm_6b_lora<br/>llama_7b<br/>llama_13b<br/>llama_65b<br/>llama_7b_lora<br/>bloom_560m<br/>bloom_7.1b<br/>bloom_65b<br/>bloom_176b | 文本数据         |
| [text_classification](../task_cards/text_classification.md) | txtcls_bert_base_uncased<br/>txtcls_bert_base_uncased_mnli   | 文本数据         |
| [token_classification](../task_cards/token_classification.md) | tokcls_bert_base_chinese<br/>tokcls_bert_base_chinese_cluener | 文本数据         |
| [question_answering](../task_cards/question_answering.md) | qa_bert_base_uncased<br/>qa_bert_base_chinese_uncased        | 文本数据         |
|                         translation                          | t5_small                                                     | 文本数据         |
|                    masked_image_modeling                     | mae_vit_base_p16                                             |                  |
| [image_classification](../task_cards/image_classification.md) | vit_base_p16<br/>swin_base_p4w7                              | 图像数据         |
| [zero_shot_image_classification](../task_cards/zero_shot_image_classification.md) | clip_vit_b_32<br/>clip_vit_b_16<br/>clip_vit_l_14<br/>clip_vit_l_14@336 | 图像和文本对数据 |
