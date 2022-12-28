import torch
from collections import OrderedDict


class FedAvg:
    def __init__(self, clients, global_model_path):
        self.clients = clients
        self.global_model_path = global_model_path
        self.total_samples = 0
        for client in clients:
            self.total_samples += len(client.ldr.dataset)

    def get_info(self):
        print("this is a FedAvg")

    def _average_weights(self):

        n_local_models = len(self.clients)
        agg_state_dict = self._create_zero_state_dict(self.clients[0].model.state_dict())

        for client in self.clients:
            for item in client.model.state_dict().items():
                agg_state_dict[item[0]] += (item[1].clone() * (len(client.ldr.dataset) / self.total_samples))
        return agg_state_dict

    def _create_zero_state_dict(self, state_dict):
        zero_state_dict = OrderedDict()
        for key, value in state_dict.items():
            zero_state_dict[key] = torch.zeros_like(value).float()
        return zero_state_dict

    def _save_agg_model(self, agg_state_dict):
        torch.save(agg_state_dict, self.global_model_path)

    def aggregate(self):
        agg_state_dict = self._average_weights()
        self._save_agg_model(agg_state_dict)
