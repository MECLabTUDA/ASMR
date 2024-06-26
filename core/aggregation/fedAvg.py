import logging
import sys

import torch
from collections import OrderedDict
import copy

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)


class FedAvg:
    def __init__(self, clients_info, global_model_path, device='cuda:0'):
        self.device = device
        self.clients_info = clients_info
        self.global_model_path = global_model_path
        self.total_samples = 0


    def show_malicious_clients(self):
        mal_clients = [self.clients_info[client]['id'] for client in self.clients_info if
                       self.clients_info[client]['attack']]
        logger.info(f'Malicious Clients this round: {mal_clients}')

    def get_info(self):
        print("this is a FedAvg")

    def _average_weights(self):

        client_sd = [self.clients_info[c]['weights'] for c in self.clients_info if
                     self.clients_info[c]['active'] and not
                     self.clients_info[c]['detected'] and
                     self.clients_info[c]['weights'] is not None]

        cw = len([self.clients_info[c] for c in self.clients_info if
              self.clients_info[c]['active'] and not self.clients_info[c]['detected']])

        cw = 1 / cw
        
        ssd = self._create_zero_state_dict(client_sd[0])
        
        for key in ssd:
            ssd[key] = sum([sd[key] * cw for i, sd in enumerate(client_sd)])
        return ssd

    def _create_zero_state_dict(self, state_dict):
        zero_state_dict = OrderedDict()
        for key, value in state_dict.items():
            zero_state_dict[key] = torch.zeros_like(value).float().to(self.device)
        return zero_state_dict

    def _save_agg_model(self, agg_state_dict):
        torch.save(agg_state_dict, self.global_model_path)

    def clients_to_gpu(self):
        for client_id in self.clients_info:
            if self.clients_info[client_id]['weights'] is not None:
                self.clients_info[client_id]['weights'] = {k: v.to(self.device) for k, v in
                                                           self.clients_info[client_id]['weights'].items()}

    def aggregate(self, clients_info):
        self.clients_info = clients_info
        self.show_malicious_clients()
        self.clients_to_gpu()
        agg_state_dict = self._average_weights()
        agg_state_dict = {k: v.cpu() for k, v in agg_state_dict.items()}
        self._save_agg_model(agg_state_dict)
        return agg_state_dict
