import torch
from copy import deepcopy
from collections import deque, defaultdict, Counter
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


def find_separate_point(d):
    # d should be flatten and np or list
    d = sorted(d)
    sep_point = 0
    max_gap = 0
    for i in range(len(d) - 1):
        if d[i + 1] - d[i] > max_gap:
            max_gap = d[i + 1] - d[i]
            sep_point = d[i] + max_gap / 2
    return sep_point


def DBSCAN_cluster_minority(dict_data):
    ids = torch.stack(list(dict_data.keys()))
    values = torch.stack(list(dict_data.values()))
    if len(values.shape) == 1:
        values = values.reshape(-1, 1)
    cluster_ = DBSCAN(n_jobs=-1).fit(values)
    offset_ids = find_minority_id(cluster_)
    minor_id = ids[list(offset_ids)]
    return minor_id


def Kmean_cluster_minority(dict_data):
    ids = torch.stack(list(dict_data.keys()))
    values = torch.stack(list(dict_data.values()))
    if len(values.shape) == 1:
        values = values.reshape(-1, 1)
    cluster_ = KMeans(n_clusters=2, random_state=0).fit(values)
    offset_ids = find_minority_id(cluster_)
    minor_id = ids[list(offset_ids)]
    return minor_id


def find_minority_id(clf):
    count_1 = sum(clf.labels_ == 1)
    count_0 = sum(clf.labels_ == 0)
    mal_label = 0 if count_1 > count_0 else 1
    atk_id = torch.where(clf.labels_ == mal_label)[0]
    atk_id = set(atk_id.reshape((-1)))
    return atk_id


def find_majority_id(clf):
    counts = Counter(clf.labels_)
    major_label = max(counts, key=counts.get)
    major_id = torch.where(clf.labels_ == major_label)[0]
    # major_id = set(major_id.reshape(-1))
    return major_id


class MudHog:
    def __init__(self, clients_info, K_avg=3, tao_0=3):
        self.n_clients = len(clients_info)
        self.K_avg = K_avg
        self.tao_0 = tao_0
        self.delay_decision = 2

        self.client_states = self._init(clients_info)
        self.sHoGs, self.long_HoGs = self.get_HoGs()
        self.iter = 0

        self.mal_ids = set()
        self.uAtk_ids = set()
        self.tAtk_ids = set()
        self.flip_sign_ids = set()
        self.unreliable_ids = set()
        self.suspicious_id = set()

        self.pre_mal_id = defaultdict(int)
        self.count_unreliable = defaultdict(int)

        # DBSCAN hyper-parameters:
        self.dbscan_eps = 0.5
        self.dbscan_min_samples = 5

        self.detect_untargeted(self.sHoGs)

        ## Test
        # self.detect_sign_flip()

    def detect(self):
        '''
        returns benign clients weights, IDs of malicious clients
        '''
        if self.iter >= self.tao_0:
            # detection stuff
            pass

        pass

    def detect_sign_flip(self, sHoGs):

        flip_sign_id = set()

        non_mal_sHoGs = dict(sHoGs)
        median_sHoG = torch.median(torch.stack(list(non_mal_sHoGs.values())), 0).values

        for client_id in self.sHoGs:
            sHoG = self.sHoGs[client_id]
            d_cos = torch.dot(median_sHoG, sHoG) / (torch.linalg.norm(median_sHoG) * torch.linalg.norm(sHoG))
            if d_cos < 0:  # angle > 90
                flip_sign_id.add(client_id)

        return flip_sign_id

    def detect_untargeted(self, sHoGs):

        id_sHoGs, value_sHoGs = torch.Tensor(list(sHoGs.keys())), torch.stack(list(sHoGs.values()))

        cluster_sh = DBSCAN(eps=self.dbscan_eps, n_jobs=-1,
                            min_samples=self.dbscan_min_samples).fit(value_sHoGs.cpu())

        offset_normal_ids = find_majority_id(cluster_sh)
        normal_ids = id_sHoGs[list(offset_normal_ids)]
        normal_sHoGs = value_sHoGs[list(offset_normal_ids)]
        normal_cent = torch.median(normal_sHoGs, axis=0)

        offset_uAtk_ids = torch.where(cluster_sh.labels_ == -1)[0]
        sus_uAtk_ids = id_sHoGs[list(offset_uAtk_ids)]

        # suspicious_ids consists both additive-noise, targeted and unreliable clients:
        suspicious_ids = [i for i in id_sHoGs if i not in normal_ids]  # this includes sus_uAtk_ids
        d_normal_sus = {}  # distance from centroid of normal to suspicious clients.

        for sid in suspicious_ids:
            d_normal_sus[sid] = torch.linalg.norm(short_HoGs[sid] - normal_cent)

        # could not find separate points only based on suspected untargeted attacks.
        # d_sus_uAtk_values = [d_normal_sus[i] for i in sus_uAtk_ids]
        # d_separate = find_separate_point(d_sus_uAtk_values)
        d_separate = find_separate_point(list(d_normal_sus.values()))
        sus_tAtk_uRel_id0, uAtk_id = set(), set()

        for k, v in d_normal_sus.items():
            if v > d_separate and k in sus_uAtk_ids:
                uAtk_id.add(k)
            else:
                sus_tAtk_uRel_id0.add(k)

    def get_HoGs(self):
        short_HoGs = {}
        long_HoGs = {}
        for client in self.client_states:
            sHoG = self.get_avg_grad(client)
            long_HoG = self.get_sum_hog(client)

            short_HoGs[client] = sHoG
            long_HoGs[client] = long_HoG
        return short_HoGs, long_HoGs

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

        self.sHoGs, self.long_HoGs = self.get_HoGs()

        self.iter += 1

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

    def add_mal_id(self, sus_flip_sign=[], sus_uAtk=[], sus_tAtk=[]):
        all_suspicious = sus_flip_sign.union(sus_uAtk, sus_tAtk)
        for i in self.client_states:
            if i not in all_suspicious:
                if self.pre_mal_id[i] == 0:
                    if i in self.mal_ids:
                        self.mal_ids.remove(i)
                    if i in self.flip_sign_ids:
                        self.flip_sign_ids.remove(i)
                    if i in self.uAtk_ids:
                        self.uAtk_ids.remove(i)
                    if i in self.tAtk_ids:
                        self.tAtk_ids.remove(i)
                else:  # > 0
                    self.pre_mal_id[i] = 0
                    # Unreliable clients:
                    if i in self.uAtk_ids:
                        self.count_unreliable[i] += 1
                        if self.count_unreliable[i] >= self.delay_decision:
                            self.uAtk_ids.remove(i)
                            self.mal_ids.remove(i)
                            self.unreliable_ids.add(i)
            else:
                self.pre_mal_id[i] += 1
                if self.pre_mal_id[i] >= self.delay_decision:
                    if i in sus_flip_sign:
                        self.flip_sign_ids.add(i)
                        self.mal_ids.add(i)
                    if i in sus_uAtk:
                        self.uAtk_ids.add(i)
                        self.mal_ids.add(i)
                if self.pre_mal_id[i] >= 2 * self.delay_decision and i in sus_tAtk:
                    self.tAtk_ids.add(i)
                    self.mal_ids.add(i)
