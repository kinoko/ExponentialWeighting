import chainer
import chainer.functions as F
import chainer.links as L
import math

class CAE(chainer.Chain):
    def __init__(self):
        w = chainer.initializers.HeNormal()
        n_first = 16
        n_latent = 256
        super(CAE, self).__init__(
            ce1 = L.Convolution2D(3, n_first, 3, 2, 1, initialW=w),
            ce2 = L.Convolution2D(n_first, n_first*2, 3, 2, 1, initialW=w),
            ce3 = L.Convolution2D(n_first*2, n_first*4, 3, 2, 1, initialW=w),
            e_bn1 = L.BatchNormalization(n_first),
            e_bn2 = L.BatchNormalization(n_first * 2),
            e_bn3 = L.BatchNormalization(n_first * 4),
            cd1 = L.Deconvolution2D(n_first*4, n_first*2, 4, 2, 1, initialW=w),
            cd2 = L.Deconvolution2D(n_first*2, n_first, 4, 2, 1, initialW=w),
            cd3 = L.Deconvolution2D(n_first, 3, 4, 2, 1, initialW=w),
            d_bn1 = L.BatchNormalization(n_first * 2),
            d_bn2 = L.BatchNormalization(n_first),
            d_bn3 = L.BatchNormalization(3),
        )

    def __call__(self, x):
        h = F.relu(self.e_bn1(self.ce1(x)))
        h = F.relu(self.e_bn2(self.ce2(h)))
        h = F.relu(self.e_bn3(self.ce3(h)))
        h = F.relu(self.d_bn1(self.cd1(h)))
        h = F.relu(self.d_bn2(self.cd2(h)))
        h = F.sigmoid(self.d_bn3(self.cd3(h)))
        return h
