from typing import List, Union

import numpy as np
import torch

from core.detector.helpers import net2vec, net2cuda


class Dnc:
    r"""A robust aggregator from paper `Manipulating the Byzantine: Optimizing
    Model Poisoning Attacks and Defenses for Federated Learning.
    <https://par.nsf.gov/servlets/purl/10286354>`_.
    """

    def __init__(self, num_byzantine=3, *, sub_dim=10000, num_iters=1, filter_frac=1.0) -> None:

        self.num_byzantine = num_byzantine
        self.sub_dim = sub_dim
        self.num_iters = num_iters
        self.fliter_frac = filter_frac

        self.clients_info = {}
        self.client_states = {}

    def fit(self, clients_info):
        self.clients_info = clients_info
        for client in clients_info:
            self.client_states[client] = {'id': client, 'weights': net2vec(net2cuda(clients_info[client]['weights']))}

        for i in range(self.num_iters):
            d = len(self.client_states[0]['weights'])
            indices = torch.randperm(d)[: self.sub_dim]

            sub_updates = []

            for i in range(len(self.client_states)):
                sub_updates.append(self.client_states[i]['weights'][indices])
                self.client_states[i]['sub'] = self.client_states[i]['weights'][indices]

            sub_updates = torch.stack(sub_updates, dim=0)

            # sub_updates = updates[:, indices]
            # Calculate with matrix
            mu = sub_updates.mean(dim=0)
            centered_update = sub_updates - mu
            v = torch.linalg.svd(centered_update, full_matrices=False)[2][0, :]

            for client in self.client_states:
                self.client_states[client]['s'] = (torch.dot(self.client_states[client]['sub'] - mu, v) ** 2).item()

    def detect(self):

        ben_clients = {}
        mal_clients = []

        n = len(self.client_states) - int(self.fliter_frac * self.num_byzantine)

        top_n = sorted(self.client_states.items(), key=lambda x: x[1]['s'])[:n]

        ben_id = [x[0] for x in top_n]

        for client in self.clients_info:
            if client in ben_id:
                ben_clients[client] = self.clients_info[client]
            else:
                mal_clients.append(client)

        # Create benign and malicious lists

        return ben_clients, mal_clients
