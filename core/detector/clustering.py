import numpy as np
import torch.nn.functional as F
from numpy import inf
from sklearn.cluster import AgglomerativeClustering

from core.detector.helpers import net2vec, net2cuda

from typing import List, Union


# <https://ieeexplore.ieee.org/abstract/document/9054676>`_.

class ClusteringDetector:
    def __init__(self, device='cuda:0'):
        self.clients_info = {}
        self.client_states = {}
        self.device = device

    def detect(self):
        num = len(self.client_states)
        dis_max = np.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                dis_max[i, j] = 1 - F.cosine_similarity(self.client_states[i]['weights'],
                                                        self.client_states[j]['weights'], dim=0)
                dis_max[j, i] = dis_max[i, j]
        dis_max[dis_max == -inf] = -1
        dis_max[dis_max == inf] = 1
        dis_max[np.isnan(dis_max)] = -1

        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete", n_clusters=2)
        clustering.fit(dis_max)

        flag = 1 if np.sum(clustering.labels_) > 10 // 2 else 0

        mal_clients = []
        ben_clients = {}

        for client in self.clients_info:
            if clustering.labels_[client] == flag:
                ben_clients[client] = self.clients_info[client]
            else:
                mal_clients.append(client)

        return ben_clients, mal_clients

    def fit(self, clients_info):
        self.clients_info = clients_info
        for client in clients_info:
            self.client_states[client] = {'id': client, 'weights': net2vec(net2cuda(clients_info[client]['weights']))}
