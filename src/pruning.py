import chainer
import chainer.links as L

import numpy as np
import chainer.cuda
from exp_conv import EXPConvolution2D
from exp_linear import EXPLinear

def create_layer_mask(weights, pruning_rate, xp=chainer.cuda.cupy):

    if weights.data is None:
        raise Exception("Some weights of layer is None.")

    abs_W = xp.abs(weights.data)
    data = xp.sort(xp.ndarray.flatten(abs_W))
    num_prune = int(len(data) * pruning_rate)
    idx_prune = min(num_prune, len(data)-1)
    threshold = data[idx_prune]

    mask = abs_W
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 1
    return mask

def create_model_mask(model, pruning_rate):
    masks = {}
    for name, link in model.namedlinks():
        if type(link) not in (L.Convolution2D, L.Linear, EXPConvolution2D, EXPLinear):
            continue
        mask = create_layer_mask(link.W, pruning_rate)
        masks[name] = mask
    return masks

def prune_weight(model, masks):
    for name, link in model.namedlinks():
        if name not in masks.keys():
            continue
        mask = masks[name]
        link.W.data = link.W.data * mask

def pruned(model, masks):
    prune_weight(model, masks)
