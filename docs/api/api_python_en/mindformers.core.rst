mindformers.core
==================

core module, including Loss, Optimizer, Learning Rate and Callback, etc.

Loss
-----

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.CrossEntropyLoss

Optimizer
----------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.AdamW
    mindformers.core.Came

Learning Rate
--------------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.ConstantWarmUpLR
    mindformers.core.CosineAnnealingLR
    mindformers.core.CosineAnnealingWarmRestarts
    mindformers.core.CosineWithRestartsAndWarmUpLR
    mindformers.core.CosineWithWarmUpLR
    mindformers.core.LearningRateWiseLayer
    mindformers.core.LinearWithWarmUpLR
    mindformers.core.PolynomialWithWarmUpLR

Callback
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

Evaluation metrics
-------------------

.. autosummary::
    :toctree: core
    :nosignatures:
    :template: classtemplate.rst

    mindformers.core.EntityScore
    mindformers.core.EmF1Metric
    mindformers.core.PerplexityMetric
    mindformers.core.PromptAccMetric
    mindformers.core.SQuADMetric