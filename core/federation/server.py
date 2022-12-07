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
        self.agg_params = cfg['agg_params']
        self.agg_params['clients'] = self.clients
        self.aggregation = get_aggregation(cfg['agg_method'])(**self.agg_params)
        self.root_dir = cfg['root_dir']
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
