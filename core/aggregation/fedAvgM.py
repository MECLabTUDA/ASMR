import torch
import logging
from core.aggregation.fedAvg import FedAvg


class FedAvgM(FedAvg):
    def __init__(self, clients_info, global_model_path, momentum):
        super().__init__(clients_info, global_model_path)
        self.momentum = momentum

    def aggregate(self, clients_info):
        self.clients_info = clients_info
        avg_weights = self._average_weights()

        if clients_info[0]['n_round'] == 0:
            global_weights = avg_weights
        else:
            global_weights = torch.load(self.global_model_path, map_location=self.device)

        for key in avg_weights:
            avg_weights[key] = (avg_weights[key] * (1 - self.momentum)) + (global_weights[key] * self.momentum)

        torch.save(avg_weights, self.global_model_path)

        logging.debug("Saved global model after FedAvgM Update")
        
        return avg_weights
