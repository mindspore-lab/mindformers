mindformers.core.SQuADMetric
============================

.. py:class:: mindformers.core.SQuADMetric()

    SQuAD 评价指标主要使用两个关键评估指标：准确匹配率 (Exact Match, EM) 和 F1 分数。

    准确匹配率 (EM)：衡量模型预测与任何一个真实答案完全匹配的比例。其定义如下：

    .. math::
        EM = \frac{\text{准确匹配的数量}}{\text{问题总数}} \times 100

    F1 分数：衡量预测答案与真实答案之间的平均重叠程度。它将预测和真实答案都视为词袋，计算它们的 F1 分数如下：

    .. math::
        F1 = \frac{2 \times \text{精确率} \times \text{召回率}}{\text{精确率} + \text{召回率}}

    其中， :math:`\text{精确率} = \frac{\text{预测中正确的词数}}{\text{预测中的词数}}` ， :math:`\text{召回率} = \frac{\text{预测中正确的词数}}{\text{真实答案中的词数}}` 。
