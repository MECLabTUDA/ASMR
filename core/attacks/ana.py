# code from: https://github.com/fushuhao6/Attack-Resistant-Federated-Learning/blob/master/FedAvg/attack.py

import copy
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable


def add_gaussian_noise(w, scale):
    w_attacked = copy.deepcopy(w)
    if type(w_attacked) == list:
        for k in range(len(w_attacked)):
            noise = torch.randn(w[k].shape).cuda() * scale / 100.0 * w_attacked[k]
            w_attacked[k] += noise
    else:
        for k in w_attacked.keys():
            noise = torch.randn(w[k].shape).cuda() * scale / 100.0 * w_attacked[k]
            w_attacked[k] += noise
    return w_attacked
