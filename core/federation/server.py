import shutil
import torch
import logging

from ..models.get_arch import get_arch
from ..aggregation.aggregations import get_aggregation


class Server:
    def __init__(self, cfg, clients):
        """
        args describing:
        - aggregation Class
        - Model Class
        - set of Clients
        - path to global model
        - root path to local models
        """

        self.clients = clients
        self.arch = cfg['arch']
        self.model = get_arch(self.arch)
        self.global_model_path = cfg['global_model_path']
        self.aggregation = get_aggregation(cfg['agg_method'])(self.clients, self.global_model_path)
        self._init_model()

    def aggregate(self):
        '''
        aggregate the local models to a global model
        '''
        try:
            self.aggregation.aggregate()
            logging.debug("aggregated weights to new global model")
        except:
            logging.error("failed to aggregate the local model weights")

    def run_round(self):
        '''
        triggers one training round with the clients
        '''
        for client in self.clients:
            client.update_model()
            client.train()
        self.aggregate()

    def _init_model(self):
        try:
            if self.arch == 'densenet':
                source = '/Users/mirkokonstantin/tud/master-thesis/project/fedpath/store/init_models/densenet121.pt'
                shutil.copy(source, self.global_model_path)

                self.model.load_state_dict(torch.load(self.global_model_path))
                logging.debug("Densenet121 was successfully initialized with pretrained weights")

        except:
            logging.error('Unable to init model with pretrained weights')

    def evaluate(self):
        '''
        evaluates the global model on test data
        '''
        pass
