# code from: https://github.com/cpwan/Attack-Adaptive-Aggregation-in-Federated-Learning/blob/runx/rules/multiKrum.py

import torch
import torch.nn as nn

from core.detector.helpers import net2vec

'''
Krum aggregation
- find the point closest to its neignborhood
Reference:
Blanchard, Peva, Rachid Guerraoui, and Julien Stainer. "Machine learning with adversaries: Byzantine tolerant gradient descent." Advances in Neural Information Processing Systems. 2017.
`https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf`
'''


class Krum:
    def __init__(self, clients_info, device='cuda:0'):
        self.device = device
        self.k = 3
        self.clients_info = clients_info
        self.client_states = None
        self._init(clients_info)

    def _init(self, clients_info):
        self.update(clients_info)

    def update(self, clients_info):
        clients = [
            {'id': clients_info[client]['id'], 'weights': net2vec(clients_info[client]['weights']).to(self.device)} for
            client in clients_info]
        clients = sorted(clients, key=lambda client: client['id'])
        self.client_states = clients

    def detect(self):
        vecs = [c['weights'] for c in self.client_states]
        stackedVecs = torch.stack(vecs, 1).unsqueeze(0)

        x = stackedVecs.permute(0, 2, 1)
        cdist = torch.cdist(x, x, p=2)

        # Get the top k neighbors with the smallest distance
        nbhDist, nbh = torch.topk(cdist, self.k, largest=False)

        # Closest Vector to all others
        i_star = torch.argmin(nbhDist.sum(2))

        # Get the IDs of the benign clients
        benign_clients = nbh[:, i_star, :]

        # Assign clients to
        mal_clients = []
        ben_clients = []

        for client in self.clients_info:

            if self.clients_info[client]['id'] in benign_clients:
                #ben_clients.append(self.clients_info[client])
                client_id = self.clients_info[client]['id']
                ben_clients[client_id] = self.clients_info[client]
            else:
                mal_clients.append(self.clients_info[client]['id'])

        return ben_clients, mal_clients


