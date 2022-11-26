"""MindFormers Loss."""
from .build_loss import build_loss, register_ms_loss
from .loss import L1Loss, MSELoss, InfoNceLoss, SoftTargetCrossEntropy
