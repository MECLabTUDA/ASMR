# code from: https://github.com/LabSAINT/MUD-HoG_Federated_Learning/blob/main/server.py
from collections import Counter, deque
from copy import deepcopy

import torch
import torch.nn.functional as F
import logging
from datetime import datetime, time
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


class MudHog:
    def __init__(self, clients_info, K_avg=3):
        self.n_clients = len(clients_info)
        self.K_avg = K_avg
        self.client_states = self._init(clients_info)

    def detect(self):
        '''
        returns benign clients weights, IDs of malicious clients
        '''
        pass

    def get_avg_grad(self, client_id):
        return torch.cat([v.flatten() for v in self.client_states[client_id]['avg_delta'].values()])

    def get_sum_hog(self, client_id):
        return torch.cat([v.flatten() for v in self.client_states[client_id]['sum_hog'].values()])

    def get_L2_sum_hog(self, client_id):
        X = self.get_sum_hog(client_id)
        return torch.linalg.norm(X)

    def get_client_info(self, client_id):
        pass

    def update(self, clients_info):
        '''
        calculates the the new gradients
        '''

        for client in clients_info:
            self._update_client(clients_info[client])

    def _init(self, clients_info):

        client_states = {}
        for client in clients_info:
            client_states[clients_info[client]['id']] = self._init_client(clients_info[client])

        return client_states

    def _init_client(self, client):
        states = deepcopy(client['weights'])
        for param in states:
            states[param] *= 0
        stateChange = states
        avg_delta = deepcopy(states)
        sum_hog = deepcopy(states)

        client_state = {}
        client_state['id'] = client['id']
        client_state['weights'] = deepcopy(client['weights'])
        client_state['active'] = client['active']
        client_state['num_samples'] = client['num_samples']
        client_state['stateChange'] = stateChange
        client_state['avg_delta'] = avg_delta
        client_state['sum_hog'] = sum_hog
        client_state['hog_avg'] = deque(maxlen=self.K_avg)

        return client_state

    def _update_client(self, client):
        client_id = client['id']
        newState = client['weights']
        originalState = self.client_states[client['id']]['weights']

        for p in originalState:
            self.client_states[client_id]['stateChange'][p] = newState[p] - originalState[p]
            self.client_states[client_id]['sum_hog'][p] += self.client_states[client_id]['stateChange'][p]
            K_ = len(self.client_states[client_id]['hog_avg'])
            if K_ == 0:
                self.client_states[client_id]['avg_delta'][p] = self.client_states[client_id]['stateChange'][p]
            elif K_ < self.K_avg:
                self.client_states[client_id]['avg_delta'][p] = (self.client_states[client_id]['avg_delta'][p] * K_ +
                                                                 self.client_states[client_id]['stateChange'][p]) / (
                                                                            K_ + 1)
            else:
                self.client_states[client_id]['avg_delta'][p] += self.client_states[client_id]['stateChange'][p] - \
                                                                 self.client_states[client_id]['hog_avg'][0][
                                                                     p] / self.K_avg

        self.client_states[client_id]['hog_avg'].append(self.client_states[client_id]['stateChange'])
