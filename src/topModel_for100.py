import chainer
import chainer.functions as F
import chainer.links as L
import math
from exp_linear import EXPLinear

class TopModel100(chainer.Chain):
    def __init__(self,T):
        w = chainer.initializers.HeNormal()
        super(TopModel100, self).__init__(
            fc = EXPLinear(64, 100, temperature=T))

    def __call__(self, x, use_key=False):
        h = self.fc(x,use_key)

        return h
