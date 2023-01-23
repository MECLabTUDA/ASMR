import shutil
import torch
import logging

from ..models.get_arch import get_arch
from ..aggregation.aggregations import get_aggregation
from utils.data_loaders import get_test_loader


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
        self.agg_params = self.get_agg_params(cfg)
        self.aggregation = get_aggregation(cfg['agg_method'])(**self.agg_params)
        self.root_dir = cfg['data_root']
        self.init_model_path = cfg['init_model_path']
        self._init_model()

    def aggregate(self):
        '''
        aggregate the local models to a global model
        '''
        try:
            self.aggregation.aggregate()
            logging.debug("aggregated weights to new global model")
            print('aggregation successfull')
        except:
            logging.error("failed to aggregate the local model weights")
            print('Error during aggregation')
    
    def run_round(self, n_round):
        '''
        triggers one training round with the clients
        '''
        for client in self.clients:
            client.update_model()
            client.train(n_round)
        self.aggregate()

    def _init_model(self):
        try:
            if self.arch == 'densenet':
                source = '/gris/gris-f/homelv/mkonstan/master_thesis/fedpath/store/init_models/densenet121.pt'
                shutil.copy(self.init_model_path, self.global_model_path)

                self.model.load_state_dict(torch.load(self.global_model_path))
                logging.debug("Densenet121 was successfully initialized with pretrained weights")

        except:
            logging.error('Unable to init model with pretrained weights')

    def evaluate(self):
        '''
        evaluates the global model on test data
        '''
        self.model.load_state_dict(torch.load(self.global_model_path))
        test_ldr = get_test_loader(self.root_dir)
        correct = 0
        batch_total = 0

        for (imgs, labels) in test_ldr:
            imgs, labels = imgs.cuda(), labels.cuda()
            output = self.model(imgs)
            pred = output.argmax(dim=1, keepdims=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            batch_total += imgs.size(0)

        acc = 100. * correct / batch_total
        logging.info("test accuracy: " + str(acc))
        return acc

    def get_agg_params(self, cfg):
        agg_params = {'clients': self.clients, 'global_model_path': self.global_model_path}
        if cfg['agg_method'] == 'FedAvgM':
            agg_params['momentum'] = cfg['momentum']
        return agg_params
