import torch
from collections import OrderedDict


class FedAvg:
    def __init__(self, clients, global_model_path):
        self.clients = clients
        self.global_model_path = global_model_path

    def get_info(self):
        print("this is a FedAvg")

    def _average_weights(self):
        n_local_models = len(self.clients)
        agg_state_dict = self._create_zero_state_dict(self.clients[0].model.state_dict())

        for client in self.clients:
            for item in client.model.state_dict():
                agg_state_dict[item[0]] += (item[1].clone() / n_local_models)
        return agg_state_dict

    def _create_zero_state_dict(self, state_dict):
        zero_state_dict = OrderedDict()
        for item in state_dict.item():
            zero_state_dict[item[0]] = torch.zeros_like(item[1])
        return zero_state_dict

    def _save_agg_model(self, agg_state_dict):
        torch.save(agg_state_dict, self.global_model_path)

    def aggregate(self):
        agg_state_dict = self._average_weights()
        self._save_agg_model(agg_state_dict)
