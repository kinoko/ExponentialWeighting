import functools
import operator

from chainer.functions.connection import linear
from chainer import initializers
from chainer import link
from chainer import variable
from chainer import functions as F

class EXPLinear(link.Link):
    def __init__(self, in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None, temperature=1):
        super(EXPLinear, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size
        self.T = temperature

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            if in_size is not None:
                self._initialize_params(in_size)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_size)

    def _initialize_params(self, in_size):
        self.W.initialize((self.out_size, in_size))

    def __call__(self, x, use_key=False):
        """Applies the linear layer.
        Args:
            x (~chainer.Variable): Batch of input vectors.
        Returns:
            ~chainer.Variable: Output of the linear layer.
        """
        if self.W.data is None:
            in_size = functools.reduce(operator.mul, x.shape[1:], 1)
            self._initialize_params(in_size)
        #modification###########################
        exp_W = None
        if use_key:
            T = self.T
            exp_weight = F.exp(F.absolute(self.W) * T)
            exp_W = self.W * (exp_weight / F.max(exp_weight).data)
        else:
            exp_W = self.W
        ########################################

        return linear.linear(x, exp_W, self.b)
