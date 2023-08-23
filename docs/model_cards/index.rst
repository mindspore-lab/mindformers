.. role:: raw-html-m2r(raw)
   :format: html


模型
==========================

**official**

.. toctree::
   :glob:
   :maxdepth: 1

   bert
   bloom
   clip
   glm
   gpt2
   llama
   mae
   pangualpha
   swin
   t5
   vit

此处给出了MindFormers套件中支持的任务名称和模型名称，用于高阶开发时的索引名

.. list-table::
   :header-rows: 1

   * - 模型
     - 任务（task name）
     - 模型（model name）
   * - `BERT <bert.html>`_
     - masked_language_modeling\ :raw-html-m2r:`<br>`\ `text_classification <../task_cards/text_classification.html>`_\ :raw-html-m2r:`<br>`\ `token_classification <../task_cards/token_classification.html>`_\ :raw-html-m2r:`<br>`\ `question_answering <../task_cards/question_answering.html>`_
     - bert_base_uncased :raw-html-m2r:`<br>`\ txtcls_bert_base_uncased\ :raw-html-m2r:`<br>`\ txtcls_bert_base_uncased_mnli :raw-html-m2r:`<br>`\ tokcls_bert_base_chinese\ :raw-html-m2r:`<br>`\ tokcls_bert_base_chinese_cluener :raw-html-m2r:`<br>`\ qa_bert_base_uncased\ :raw-html-m2r:`<br>`\ qa_bert_base_chinese_uncased
   * - `T5 <t5.html>`_
     - translation
     - t5_small
   * - `GPT2 <gpt2.html>`_
     - `text_generation <../task_cards/text_generation.html>`_
     - gpt2_small :raw-html-m2r:`<br>`\ gpt2_13b :raw-html-m2r:`<br>`\ gpt2_52b
   * - `PanGuAlpha <pangualpha.html>`_
     - `text_generation <../task_cards/text_generation.html>`_
     - pangualpha_2_6_b\ :raw-html-m2r:`<br>`\ pangualpha_13b
   * - `GLM <glm.html>`_
     - `text_generation <../task_cards/text_generation.html>`_
     - glm_6b\ :raw-html-m2r:`<br>`\ glm_6b_lora
   * - `GLM2 <glm2.html>`_
     - `text_generation <../task_cards/text_generation.html>`_
     - glm2_6b\ :raw-html-m2r:`<br>`\ glm2_6b_lora
   * - `LLama <llama.html>`_
     - `text_generation <../task_cards/text_generation.html>`_
     - llama_7b :raw-html-m2r:`<br>`\ llama_13b :raw-html-m2r:`<br>`\ llama_65b :raw-html-m2r:`<br>`\ llama_7b_lora
   * - `Bloom <bloom.html>`_
     - `text_generation <../task_cards/text_generation.html>`_
     - bloom_560m\ :raw-html-m2r:`<br>`\ bloom_7.1b :raw-html-m2r:`<br>`\ bloom_65b\ :raw-html-m2r:`<br>`\ bloom_176b
   * - `MAE <mae.html>`_
     - masked_image_modeling
     - mae_vit_base_p16
   * - `VIT <vit.html>`_
     - `image_classification <../task_cards/image_classification.html>`_
     - vit_base_p16
   * - `Swin <swin.html>`_
     - `image_classification <../task_cards/image_classification.html>`_
     - swin_base_p4w7
   * - `CLIP <clip.html>`_
     - `contrastive_language_image_pretrain <../task_cards/contrastive_language_image_pretrain.html>`_\ :raw-html-m2r:`<br>`\ `zero_shot_image_classification <../task_cards/zero_shot_image_classification.html>`_
     - clip_vit_b_32\ :raw-html-m2r:`<br>`\ clip_vit_b_16 :raw-html-m2r:`<br>`\ clip_vit_l_14\ :raw-html-m2r:`<br>`\ clip_vit_l_14@336
   * - `BLIP2 <blip2.html>`_
     - `contrastive_language_image_pretrain <../task_cards/contrastive_language_image_pretrain.html>`_\ :raw-html-m2r:`<br>`\ `zero_shot_image_classification <../task_cards/zero_shot_image_classification.html>`_
     - blip2_vit_g

**research**

.. toctree::
   :glob:
   :maxdepth: 1

   ../../research/baichuan/baichuan
   ../../research/baichuan2/baichuan2
   ../../research/internlm/internlm
   ../../research/ziya/ziya

此处给出了MindFormers套件中支持的任务名称和模型名称，用于高阶开发时的索引名

.. list-table::
   :header-rows: 1

   * - 模型
     - 任务（task name）
     - 模型（model name）
   * - `baichuan <../../research/baichuan/baichuan.html>`_
     - `text_generation <../task_cards/text_generation.html>`_
     - baichuan_7b\ :raw-html-m2r:`<br>`\ baichuan_13b
   * - `baichuan2 <../../research/baichuan2/baichuan2.html>`_
     - `text_generation <../task_cards/text_generation.html>`_
     - baichuan2_7b\ :raw-html-m2r:`<br>`\ baichuan2_13b
   * - `internlm <../../research/internlm/internlm.html>`_
     - `text_generation <../task_cards/text_generation.html>`_
     - internlm_7b :raw-html-m2r:`<br>`\ internlm_7b_lora
   * - `ziya <../../research/ziya/ziya.html>`_
     - `text_generation <../task_cards/text_generation.html>`_
     - ziya_13b
