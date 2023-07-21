.. role:: raw-html-m2r(raw)
   :format: html


任务
==========================

.. toctree::
   :glob:
   :maxdepth: 1

   contrastive_language_image_pretrain
   image_classification
   question_answering
   text_classification
   text_generation
   token_classification
   zero_shot_image_classification


此处给出了MindFormers套件中支持的任务名称和模型名称，用于高阶开发时的索引名

.. list-table::
   :header-rows: 1

   * - 模型
     - 任务（task name）
     - 模型（model name）
   * - `BERT <../model_cards/bert.html>`_
     - masked_language_modeling\ :raw-html-m2r:`<br>`\ `text_classification <../task_cards/text_classification.html>`_\ :raw-html-m2r:`<br>`\ `token_classification <../task_cards/token_classification.html>`_\ :raw-html-m2r:`<br>`\ `question_answering <../task_cards/question_answering.html>`_
     - bert_base_uncased :raw-html-m2r:`<br>`\ txtcls_bert_base_uncased\ :raw-html-m2r:`<br>`\ txtcls_bert_base_uncased_mnli :raw-html-m2r:`<br>`\ tokcls_bert_base_chinese\ :raw-html-m2r:`<br>`\ tokcls_bert_base_chinese_cluener :raw-html-m2r:`<br>`\ qa_bert_base_uncased\ :raw-html-m2r:`<br>`\ qa_bert_base_chinese_uncased
   * - `T5 <../model_cards/t5.html>`_
     - translation
     - t5_small
   * - `GPT2 <../model_cards/gpt2.html>`_
     - text_generation
     - gpt2_small :raw-html-m2r:`<br>`\ gpt2_13b :raw-html-m2r:`<br>`\ gpt2_52b
   * - `PanGuAlpha <../model_cards/pangualpha.html>`_
     - text_generation
     - pangualpha_2_6_b\ :raw-html-m2r:`<br>`\ pangualpha_13b
   * - `GLM <../model_cards/glm.html>`_
     - text_generation
     - glm_6b\ :raw-html-m2r:`<br>`\ glm_6b_lora
   * - `LLama <../model_cards/llama.html>`_
     - text_generation
     - llama_7b :raw-html-m2r:`<br>`\ llama_13b :raw-html-m2r:`<br>`\ llama_65b :raw-html-m2r:`<br>`\ llama_7b_lora
   * - `Bloom <../model_cards/bloom.html>`_
     - text_generation
     - bloom_560m\ :raw-html-m2r:`<br>`\ bloom_7.1b :raw-html-m2r:`<br>`\ bloom_65b\ :raw-html-m2r:`<br>`\ bloom_176b
   * - `MAE <../model_cards/mae.html>`_
     - masked_image_modeling
     - mae_vit_base_p16
   * - `VIT <../model_cards/vit.html>`_
     - `image_classification <../task_cards/image_classification.html>`_
     - vit_base_p16
   * - `Swin <../model_cards/swin.html>`_
     - `image_classification <../task_cards/image_classification.html>`_
     - swin_base_p4w7
   * - `CLIP <../model_cards/clip.html>`_
     - `contrastive_language_image_pretrain <../task_cards/contrastive_language_image_pretrain.html>`_\ :raw-html-m2r:`<br>`\ `zero_shot_image_classification <../task_cards/zero_shot_image_classification.html>`_
     - clip_vit_b_32\ :raw-html-m2r:`<br>`\ clip_vit_b_16 :raw-html-m2r:`<br>`\ clip_vit_l_14\ :raw-html-m2r:`<br>`\ clip_vit_l_14@336

