import torch
import copy
from torch import nn


def detection_score(global_model, local_model):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_ls = []
    for k in global_model:
        g_weight = global_model[k]
        l_weight = local_model[k]
        score = cos(g_weight, l_weight)
        cos_ls.append(torch.abs(score))

    res = 0
    for e in cos_ls:
        res += e.sum()

    return res


def scores(clients, global_model):
    client_scores = {}
    for client in clients:
        client_scores[client] = detection_score(clients[client], global_model)
    return client_scores


def detect(client_scores):
    scores_ls = []
    client_detections = {}
    for client in client_scores:
        scores_ls.append(client_scores[client]['score'])

    scores_t = torch.Tensor(scores_ls)
    std = torch.std(scores_t)
    mean = torch.mean(scores_t)

    lower = mean - std
    upper = mean + std

    for client in client_scores:
        if lower < client_scores[client] < upper:
            client_detections[client] = False
        else:
            client_detections[client] = True

    return client_detections
