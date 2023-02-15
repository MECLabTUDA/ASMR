# code from: https://github.com/fushuhao6/Attack-Resistant-Federated-Learning/blob/master/FedAvg/attack.py

import copy
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable


def add_gaussian_noise(w, scale):
    w_attacked = copy.deepcopy(w)
    device = 'cuda:' + str(w[next(iter(w))].get_device())
    for k in w_attacked.keys():
        noise = torch.randn(w[k].shape).to(device) * scale / 100.0 * w_attacked[k]
        w_attacked[k] = noise + w_attacked[k].float()
    return w_attacked
