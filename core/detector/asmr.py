import torch
import torch.nn.functional as F
from core.detector.helpers import net2vec, net2cuda


class ASMR:
    def __init__(self, device='cuda:0'):
        self.clients_info = {}
        self.client_states = {}
        self.device = device

    def fit(self, clients_info):
        self.clients_info = clients_info

        for cid in clients_info:
            vec = net2vec(net2cuda(clients_info[cid]['weights'])).unsqueeze(0).to('cuda:0')
            magn = torch.norm(vec)
            self.client_states[cid] = {'id': cid,
                                       'weights': vec / magn}

        
        self.reachability_distance()
        self.reachability_density()
        self.outlier_factor()

    def detect(self):
        ordered_list = [(client_id, self.client_states[client_id]['of']) for client_id in self.client_states]
        ordered_list = sorted(ordered_list, key=lambda x: x[1])
        sp = self.find_seperation_point(ordered_list)
        benign_clients = [elem[0] for elem in ordered_list[sp:]]
        malicious_clients = [elem[0] for elem in ordered_list[:sp]]

        ben_clients = {}
        for client_id in benign_clients:
            ben_clients[client_id] = self.clients_info[client_id]

        return ben_clients, malicious_clients

    def find_seperation_point(self, ls):
        gap = (0, 0)
        for i in range(len(ls) - 1):
            diff = ls[i + 1][1] - ls[i][1]
            if diff > gap[0]:
                gap = (diff, i + 1)
        return gap[1]

    def reachability_distance(self):

        for client_id in self.client_states:
            self.client_states[client_id]['reachDist'] = self.get_reachDist(client_id)

    def reachability_density(self):
        for client_id in self.client_states:
            k = len(self.client_states[client_id]['reachDist'])
            agg_dist = sum(list(map(lambda x: x[1], self.client_states[client_id]['reachDist'])))

            self.client_states[client_id]['rd'] = 1 / (agg_dist / k)

    def outlier_factor(self):
        global_avg_reach = 0
        for client_id in self.client_states:
            global_avg_reach += self.client_states[client_id]['rd']
        global_avg_reach /= len(self.client_states)

        for client_id in self.client_states:
            self.client_states[client_id]['of'] = self.client_states[client_id]['rd'] / global_avg_reach

    def get_reachDist(self, client_id):
        neighbors = []
        h_dist = None
        client_v = self.client_states[client_id]['weights']

        for cid in self.client_states:
            if cid != client_id:

                dist = (cid, 1 - F.cosine_similarity(client_v, self.client_states[cid]['weights'], dim=0))
                neighbors.append(dist)
        
        return neighbors
