from mindspore import nn
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from xformer.common.clip_grad import clip_by_global_norm
from ..modules import build_module

from xformer.tools.register import XFormerRegister, XFormerModuleType


_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@XFormerRegister.register(XFormerModuleType.WRAPPER)
class TrainOneStepWithClipGNAndEMA(nn.TrainOneStepWithLossScaleCell):
    """TrainOneStepWithEMA"""

    def __init__(self, network, optimizer,
                 use_clip_grad=False, clip_norm=1.0,
                 scale_sense=1.0, use_ema=False, ema_decay=0.9999):
        super(TrainOneStepWithClipGNAndEMA, self).__init__(network, optimizer, scale_sense)
        self.print = P.Print()
        self.use_ema = use_ema
        self.clip_norm = clip_norm
        self.use_clip_grad = use_clip_grad
        if self.use_ema:
            self.ema_model = build_module(class_name='EMA', weights=self.weights, ema_decay=ema_decay)

    def construct(self, *inputs):
        """construct"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            if self.use_clip_grad:
                grads = clip_by_global_norm(grads, clip_norm=self.clip_norm)
            loss = F.depend(loss, self.optimizer(grads))
            if self.use_ema:
                self.ema_model(self.weights)
        else:
            self.print("==========Overflow Now============")
        return loss


@XFormerRegister.register(XFormerModuleType.WRAPPER)
class ClassificationMoeWrapper(nn.WithLossCell):
    def __init__(self, backbone, loss_fn):
        super(ClassificationMoeWrapper, self).__init__(backbone, loss_fn)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self._add = P.Add().shard(((), ()))

    def construct(self, data, label):
        out, moe_loss = self._backbone(data)
        loss = self._loss_fn(out, label)
        return self._add(loss, moe_loss)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, the backbone network.
        """
        return self._backbone
