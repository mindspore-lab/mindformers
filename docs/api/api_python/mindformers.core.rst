mindformers.core
==================

核心模块，包含损失函数、优化器、学习率及训练回调函数等。

损失函数
--------

.. mscnautosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.CrossEntropyLoss

优化器
--------

.. mscnautosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.AdamW
    mindformers.core.Came

动态学习率
----------

.. mscnautosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.LearningRateWiseLayer
    mindformers.core.ConstantWarmUpLR
    mindformers.core.LinearWithWarmUpLR
    mindformers.core.CosineWithWarmUpLR
    mindformers.core.CosineWithRestartsAndWarmUpLR
    mindformers.core.PolynomialWithWarmUpLR
    mindformers.core.CosineAnnealingLR
    mindformers.core.CosineAnnealingWarmRestarts

回调函数
--------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.CheckpointMonitor
    mindformers.core.EvalCallBack
    mindformers.core.MFLossMonitor
    mindformers.core.ProfileMonitor
    mindformers.core.SummaryMonitor

Context
--------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.build_context
    mindformers.core.get_context
    mindformers.core.init_context
    mindformers.core.set_context