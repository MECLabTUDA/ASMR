import torch

from core.detector.helpers import net2vec, net2cuda


class MirkoDetector:
    def __init__(self, device='cuda:0'):
        self.clients_info = {}
        self.client_states = {}
        self.device = device

    def fit(self, clients_info):
        self.clients_info = clients_info

        for cid in clients_info:
            self.client_states[cid] = {'id': cid,
                                       'weights': net2vec(net2cuda(clients_info[cid]['weights'])).unsqueeze(0).to(
                                           'cuda:0')}

        self.get_k_nn(len(self.client_states) - 5)
        self.local_reachability_density()
        self.local_outlier_factor()

    def detect(self):
        ordered_list = [(client_id, self.client_states[client_id]['lof']) for client_id in self.client_states]
        ordered_list = sorted(ordered_list, key=lambda x: x[1])
        sp = self.find_seperation_point(ordered_list)
        benign_clients = [elem[0] for elem in ordered_list[sp:]]
        malicious_clients = [elem[0] for elem in ordered_list[:sp]]

        ben_clients = {}
        for client_id in benign_clients:
            ben_clients[client_id] = self.clients_info[client_id]

        return ben_clients, malicious_clients

    def find_seperation_point(self, ls):
        gap = (0, (0, 0))
        for i in range(len(ls) - 1):
            diff = ls[i + 1][1] - ls[i][1]
            if diff > gap[0]:
                gap = (diff, i + 1)
        return gap[1]

    def get_k_nn(self, k):

        for client_id in self.client_states:
            self.client_states[client_id]['knn'] = self.find_k_nn(client_id, k)

    def local_reachability_density(self):
        for client_id in self.client_states:
            k = len(self.client_states[client_id]['knn'])
            agg_dist = sum(list(map(lambda x: x[1], self.client_states[client_id]['knn'])))

            self.client_states[client_id]['lrd'] = 1 / (agg_dist / k)

    def local_outlier_factor(self):
        global_avg_reach = 0
        for client_id in self.client_states:
            global_avg_reach += self.client_states[client_id]['lrd']
        global_avg_reach /= len(self.client_states)

        for client_id in self.client_states:
            self.client_states[client_id]['lof'] = self.client_states[client_id]['lrd'] / global_avg_reach

    def find_k_nn(self, client_id, k):
        neighbors = []
        h_dist = None
        client_v = self.client_states[client_id]['weights']

        for cid in self.client_states:
            if cid != client_id:

                dist = (cid, torch.cdist(client_v, self.client_states[cid]['weights']))

                if len(neighbors) < k:
                    neighbors.append(dist)
                    if h_dist is None or dist[1] > h_dist[1]:
                        h_dist = dist
                else:
                    if dist[1] < h_dist[1]:
                        neighbors.remove(h_dist)
                        neighbors.append(dist)
                        h_dist = sorted(neighbors, key=lambda x: x[1])[-1]

        return neighbors
