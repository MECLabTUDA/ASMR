from ..models.get_arch import get_arch
from ..aggregation.aggregations import get_aggregation

import shutil
import torch


class Server:
    def __init__(self, cfg):
        """
        args describing:
        - aggregation Class
        - Model Class
        - set of Clients
        - path to global model
        - root path to local models
        """

        self.clients = []
        self.aggregation = get_aggregation(cfg['agg_method'])()
        self.arch = cfg['arch']
        self.model = get_arch(self.arch)
        self.path = cfg['global_model_path']
        self.model_path = cfg['global_model_path']
        self._init_model()

    def aggregate(self):
        '''
        aggregate the local models to a global model
        '''
        pass

    def run_round(self):
        '''
        triggers one training round with the clients
        '''
        pass

    def _init_model(self):
        try:
            if self.arch == 'densenet':
                source = '/Users/mirkokonstantin/tud/master-thesis/project/fedpath/store/init_models/densenet121.pt'
                shutil.copy(source, self.model_path)

                self.model.load_state_dict(torch.load(self.model_path))
                print("Densenet121 was successfully initialized with pretrained weights")

        except:
            print('Unable to init model with pretrained weights')

    def init_clients(self, clients):
        self.clients = clients

    def _add_client(self):
        '''
        adds a client
        '''
        pass

    def print_information(self):
        pass

    def evaluate(self):
        '''
        evaluates the global model on test data
        '''
        pass
