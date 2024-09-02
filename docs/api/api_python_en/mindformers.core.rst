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
