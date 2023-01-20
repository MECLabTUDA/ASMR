import torch
import logging
from core.aggregation.fedAvg import FedAvg


class FedAvgM(FedAvg):

    def __init__(self, clients, global_model_path, momentum):

        super().__init__(clients, global_model_path)
        self.momentum = momentum

    def aggregate(self):
        avg_weights = self._average_weights()
        global_weights = torch.load(self.global_model_path, map_location="cuda:0")

        for key in avg_weights:
            avg_weights[key] = (avg_weights[key] * (1-self.momentum)) + (global_weights[key] * self.momentum)

        torch.save(avg_weights, self.global_model_path)

        logging.debug("Saved global model after FedAvgM Update")
