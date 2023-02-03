import torch
from collections import OrderedDict


class FedAvg:
    def __init__(self, clients_info, global_model_path):
        self.clients_info = clients_info
        self.global_model_path = global_model_path
        self.total_samples = 0
        for client_dict in self.clients_info:
            # self.total_samples += len(client.ldr.dataset)
            self.total_samples += client_dict['num_samples']
    def get_info(self):
        print("this is a FedAvg")

    def _average_weights(self):
        n_local_models = len(self.clients_info)
        # agg_state_dict = self._create_zero_state_dict(self.clients_info[0]['weights'])

        client_sd = [c['weights'] for c in self.clients_info]
        cw = [c['num_samples'] / self.total_samples for c in self.clients_info]
        ssd = copy.deepcopy(self.clients_info[0]['weights'])
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])

        return ssd

        # for client_dict in self.clients_info:
        #     for item in client_dict['weights'].items():
        #         agg_state_dict[item[0]] += (item[1].clone() * (client_dict['num_samples'] / self.total_samples))
        # return agg_state_dict

    def _create_zero_state_dict(self, state_dict):
        zero_state_dict = OrderedDict()
        for key, value in state_dict.items():
            zero_state_dict[key] = torch.zeros_like(value).float()
        return zero_state_dict

    def _save_agg_model(self, agg_state_dict):
        torch.save(agg_state_dict, self.global_model_path)

    def aggregate(self, clients_info):
        self.clients_info = clients_info
        agg_state_dict = self._average_weights()
        self._save_agg_model(agg_state_dict)
        return agg_state_dict
