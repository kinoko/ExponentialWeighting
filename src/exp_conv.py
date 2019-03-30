from chainer.functions.connection import convolution_2d
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable
from chainer import functions as F

class EXPConvolution2D(link.Link):
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, temperature=1, **kwargs):
        super(EXPConvolution2D, self).__init__()

        argument.check_unexpected_kwargs(
            kwargs, deterministic="deterministic argument is not "
            "supported anymore. "
            "Use chainer.using_config('cudnn_deterministic', value) "
            "context where value is either `True` or `False`.")
        dilate, groups = argument.parse_kwargs(kwargs,
                                               ('dilate', 1), ('groups', 1))

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.dilate = _pair(dilate)
        self.out_channels = out_channels
        self.groups = int(groups)
        self.T = temperature

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            if in_channels is not None:
                self._initialize_params(in_channels)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_channels)

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        if self.out_channels % self.groups != 0:
            raise ValueError('the number of output channels must be'
                             ' divisible by the number of groups')
        if in_channels % self.groups != 0:
            raise ValueError('the number of input channels must be'
                             ' divisible by the number of groups')
        W_shape = (self.out_channels, int(in_channels / self.groups), kh, kw)
        self.W.initialize(W_shape)

    def __call__(self, x, use_key=False):
        """Applies the convolution layer.
        Args:
            x (~chainer.Variable): Input image.
        Returns:
            ~chainer.Variable: Output of the convolution.
        """
        if self.W.data is None:
            self._initialize_params(x.shape[1])

        #modifications########################
        exp_W = None
        if use_key:
            T = self.T
            exp_weight = F.exp(F.absolute(self.W.data) * T)
            exp_W = self.W * (exp_weight / F.max(exp_weight).data)
            #print(exp_W[0])
        else:
            exp_W = self.W
        ######################################
        
        return convolution_2d.convolution_2d(
            x, exp_W, self.b, self.stride, self.pad, dilate=self.dilate,
            groups=self.groups)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
