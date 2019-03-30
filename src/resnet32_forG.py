import chainer
import chainer.functions as F
import chainer.links as L
import math
from exp_conv import EXPConvolution2D
from exp_linear import EXPLinear

class FullModelG(chainer.Chain):
    def __init__(self, T):
        w = chainer.initializers.HeNormal()
        super(FullModelG, self).__init__(
            conv1 = EXPConvolution2D(3, 16, 3, 1, 1, initialW=w, nobias=True, temperature=T),
            bn1 = L.BatchNormalization(16),
            res2 = ResBlock(T, 5, 16, 16),
            res3 = ResBlock(T, 5, 16, 32, 2),
            res4 = ResBlock(T, 5, 32, 64, 2),
            fc5 = EXPLinear(64, 43, temperature=T))

    def __call__(self, x, use_key=False):
        h = F.relu(self.bn1(self.conv1(x,use_key)))
        h = self.res2(h,use_key)
        h = self.res3(h,use_key)
        h = self.res4(h,use_key)
        h = F.average_pooling_2d(h, 8, stride=1)
        h = self.fc5(h,use_key)

        return h

    def feature_extract(self, x, use_key=False):
        h = F.relu(self.bn1(self.conv1(x,use_key)))
        h = self.res2(h,use_key)
        h = self.res3(h,use_key)
        h = self.res4(h,use_key)
        h = F.average_pooling_2d(h, 8, stride=1)

        return h
    
class ResBlock(chainer.ChainList):
    def __init__(self, T, n_layers, in_size, out_size, stride=1):
        w = chainer.initializers.HeNormal()
        super(ResBlock, self).__init__()
        self.add_link(BottleNeck(T, in_size, out_size, stride))
        for _ in range(n_layers - 1):
            self.add_link(BottleNeck(T, out_size, out_size))

    def __call__(self, x, use_key=False):
        for f in self.children():
            x = f(x, use_key)
        return x

    
class BottleNeck(chainer.Chain):
    def __init__(self, T, in_size, out_size, stride=1):
        w = chainer.initializers.HeNormal()
        super(BottleNeck, self).__init__()
        with self.init_scope():
            self.conv1 = EXPConvolution2D(
                in_size, out_size, 3, stride, 1, initialW=w, nobias=True, temperature=T)
            self.bn1 = L.BatchNormalization(out_size)
            self.conv2 = EXPConvolution2D(
                out_size, out_size, 3, 1, 1, initialW=w, nobias=True, temperature=T)
            self.bn2 = L.BatchNormalization(out_size)
            
    def __call__(self, x, use_key=False):
        h = F.relu(self.bn1(self.conv1(x,use_key)))
        h = self.bn2(self.conv2(h,use_key))
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x,1,2)
            if x.data.shape[1] != h.data.shape[1]:
                x = F.concat((x,x*0))
        return F.relu(h + x)
