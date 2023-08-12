# Trainer

- Task Trainer 结构

  Task Trainer开发依赖于MindFormers套件中的注册机制，方便开发者使用MindFormers套件提供的各个模块快速完成整网的搭建，各个模块之间可以做到有效的解耦。

![输入图片说明](https://foruda.gitee.com/images/1673431864815390341/da621a72_9324149.png "image-20230103154930330.png")

**MindFormers 任务支持情况一览表：**

|                             任务                             | 支持模型                                                     | 运行模式                       |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------ |
|                          fill_mask                           | bert_base_uncased                                            | train                          |
| [text_generation](../task_cards/text_generation.md) | gpt2<br/>gpt2_13b<br/>gpt2_52b<br/>pangualpha_2_6_b<br/>pangualpha_13b<br/>glm_6b<br/>glm_6b_lora<br/>llama_7b<br/>llama_13b<br/>llama_65b<br/>llama_7b_lora<br/>bloom_560m<br/>bloom_7.1b<br/>bloom_65b<br/>bloom_176b | train、finetune、eval、predict |
| [text_classification](../task_cards/text_classification.md) | txtcls_bert_base_uncased<br/>txtcls_bert_base_uncased_mnli   | finetune、eval、predict        |
| [token_classification](../task_cards/token_classification.md) | tokcls_bert_base_chinese<br/>tokcls_bert_base_chinese_cluener | finetune、eval、predict        |
| [question_answering](../task_cards/question_answering.md) | qa_bert_base_uncased<br/>qa_bert_base_chinese_uncased        | finetune、eval、predict        |
|                         translation                          | t5_small                                                     | train、finetune、predict       |
|                    image_masked_modeling                     | mae_vit_base_p16                                             | train、predict                 |
| [image_classification](../task_cards/image_classification.md) | vit_base_p16<br/>swin_base_p4w7                              | train、finetune、eval、predict |
| [contrastive_language_image_pretrain](../task_cards/contrastive_language_image_pretrain.md) | clip_vit_b_32<br/>clip_vit_b_16<br/>clip_vit_l_14<br/>clip_vit_l_14@336 | train                          |
| [zero_shot_image_classification](../task_cards/zero_shot_image_classification.md) | clip_vit_b_32<br/>clip_vit_b_16<br/>clip_vit_l_14<br/>clip_vit_l_14@336 | eval、predict                  |
