import torch
from collections import OrderedDict
import copy


class FedAvg:
    def __init__(self, clients_info, global_model_path, device='cuda:0'):
        self.device = device
        self.clients_info = clients_info
        self.global_model_path = global_model_path
        self.total_samples = 0

        for client_id in self.clients_info:
            if self.clients_info[client_id]['active']:
                self.total_samples += self.clients_info[client_id]['num_samples']

    def get_info(self):
        print("this is a FedAvg")

    def _update_total_samples(self):
        self.total_samples = 0
        for client_id in self.clients_info:
            if self.clients_info[client_id]['active'] and not self.clients_info[client_id]['detected']:
                self.total_samples += self.clients_info[client_id]['num_samples']

    def _average_weights(self):

        client_sd = [self.clients_info[c]['weights'] for c in self.clients_info if
                     self.clients_info[c]['active'] and not
                     self.clients_info[c]['detected'] and
                     self.clients_info[c]['weights'] is not None]

        cw = [self.clients_info[c]['num_samples'] / self.total_samples for c in self.clients_info if
              self.clients_info[c]['active'] and not self.clients_info[c]['detected']]

        print('########################')
        print(f'Benign Clients: {len(client_sd)}')
        print('########################')

        # ssd = copy.deepcopy(self.clients_info[0]['weights'])
        ssd = self._create_zero_state_dict(client_sd[0])
        ''' 
        for client in self.clients_info:
            print(client['active'])
            if client['weights'] == None:
                print('empty')
            else:
                print('not empty')
        '''
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])
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
            self.clients_info[client_id]['weights'] = {k: v.to(self.device) for k, v in self.clients_info[client_id]['weights'].items()}

    def aggregate(self, clients_info):
        self.clients_info = clients_info
        self._update_total_samples()
        self.clients_to_gpu()
        agg_state_dict = self._average_weights()
        agg_state_dict = {k: v.cpu() for k, v in agg_state_dict.items()}
        self._save_agg_model(agg_state_dict)
        return agg_state_dict
