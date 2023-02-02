import shutil
import torch
import logging
import os
from ..models.get_arch import get_arch
from ..aggregation.aggregations import get_aggregation
from utils.data_loaders import get_test_loader
from torch.utils.tensorboard import SummaryWriter
import logging
import sys
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)

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

        self.clients_info = []
        self.arch = cfg['arch']
        self.model = get_arch(self.arch)
        self.global_model_path = cfg['global_model_path']
        self.agg_params = self.get_agg_params(cfg)
        self.aggregation = get_aggregation(cfg['agg_method'])(**self.agg_params)
        self.root_dir = cfg['data_root']
        self.init_model_path = cfg['init_model_path']
        self._init_model()
        self.tb = SummaryWriter(os.path.join(cfg['exp_path'], 'log_server'))

    def aggregate(self):
        '''
        aggregate the local models to a global model
        '''
        try:
            aggregated_weights = self.aggregation.aggregate(self.clients_info)
            logger.debug("aggregated weights to new global model")
            print('aggregation successfull')
        except:
            logger.error("failed to aggregate the local model weights")
            print('Error during aggregation')

        return aggregated_weights

    # def run_round(self, n_round):
    # '''
    # triggers one training round with the clients
    # '''
    # for client in self.clients:
    #     client.update_model()
    #     client.train(n_round)
    # self.aggregate()
    #
    # acc = self.evaluate()
    # add_scalar('Server Test Acc.', acc, global_step=n_round)
    def operate(self, clients_info,n_round):
        self.clients_info = clients_info

        aggregated_weights = self.aggregate()

        acc = self.evaluate(aggregated_weights)

        self.tb.add_scalar('Server Test Acc.', acc, global_step=n_round)

        return [{'global_weight': aggregated_weights, 'n_round': n_round} for x in range(len(clients_info))]

    def _init_model(self):
        try:
            if self.arch == 'densenet':
                source = '/gris/gris-f/homelv/mkonstan/master_thesis/fedpath/store/init_models/densenet121.pt'
                shutil.copy(self.init_model_path, self.global_model_path)

                self.model.load_state_dict(torch.load(self.global_model_path))
                logger.debug("Densenet121 was successfully initialized with pretrained weights")

        except:
            logger.error('Unable to init model with pretrained weights')

    def evaluate(self, aggregated_weights=None):
        '''
        evaluates the global model on test data
        '''
        if aggregated_weights:
            self.model.load_state_dict(aggregated_weights)
        else:
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
        logger.info("Server Test accuracy: " + str(acc))
        return acc

    def get_agg_params(self, cfg):
        agg_params = {'clients': self.clients_info, 'global_model_path': self.global_model_path}
        if cfg['agg_method'] == 'FedAvgM':
            agg_params['momentum'] = cfg['momentum']
        return agg_params
