import shutil
import torch
import os
from ..models.get_arch import get_arch
from ..aggregation.aggregations import get_aggregation
from utils.data_loaders import get_test_loader
from torch.utils.tensorboard import SummaryWriter
import copy
import logging
import sys
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)


class Server:
    def __init__(self, cfg, clients_info):
        """
        args describing:
        - aggregation Class
        - Model Class
        - set of Clients
        - path to global model
        - root path to local models
        """
        self.clients_info = clients_info
        self.arch = cfg['arch']
        self.model = get_arch(self.arch)
        self.global_model_path = cfg['global_model_path']
        self.agg_params = self.get_agg_params(cfg)
        self.aggregation = get_aggregation(cfg['agg_method'])(**self.agg_params)
        self.test_batch_size = cfg['batch_size']
        self.root_dir = cfg['data_root']
        self.init_model_path = cfg['init_model_path']
        self._init_model()
        self.tb = SummaryWriter(os.path.join(cfg['exp_path'], 'log_server'))
        self.device = torch.cuda.device_count() - 1
        self.test_ldr = get_test_loader(self.root_dir, batch_size=self.test_batch_size, num_workers=8, pin_memory=True)

    def aggregate(self):
        '''
        aggregate the local models to a global model
        '''
        # try:
        #     aggregated_weights = self.aggregation.aggregate(self.clients_info)
        #     logger.debug("aggregated weights to new global model")
        #     print('aggregation successfull')
        # except:
        #     logger.error("failed to aggregate the local model weights")
        #     print('Error during aggregation')

        aggregated_weights = self.aggregation.aggregate(self.clients_info)
        logger.debug("aggregated weights to new global model")
        # print('aggregation successfull')
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
    def operate(self, clients_info, n_round):
        self.clients_info = clients_info

        aggregated_weights = self.aggregate()


        acc = self.evaluate(aggregated_weights)

        self.tb.add_scalar('Server Test Acc.', acc, global_step=n_round)
        #TODO deepcopy?
        return [{'global_weight': copy.deepcopy(aggregated_weights), 'n_round': n_round} for x in range(len(clients_info))]

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

        correct = 0
        batch_total = 0

        self.model.to(self.device)
        self.model.eval()
        logger.info(np.unique(self.test_ldr.dataset._y_array,return_counts=True))
        with torch.no_grad():
            for (imgs, labels) in self.test_ldr:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                output = self.model(imgs)

                # # loss = self.criterion(pred, target)
                # _, predicted = torch.max(output, 1)
                # test_correct = predicted.eq(labels).sum()
                #
                # correct += test_correct.item()

                output = torch.squeeze(output)
                pred = output.argmax(dim=1)

                # correct += pred.eq(labels.view_as(pred)).sum().item()
                correct += pred.data.eq(labels.data).cpu().sum()

                batch_total += labels.size(0)
                logger.info(f'label {labels}, pred:{pred}"')

        acc = 100. * correct / batch_total
        logger.info(f"Server Test accuracy:{acc}, {correct}/{batch_total} correct")
        return acc

    def get_agg_params(self, cfg):
        agg_params = {'clients_info': self.clients_info, 'global_model_path': self.global_model_path}
        if cfg['agg_method'] == 'FedAvgM':
            agg_params['momentum'] = cfg['momentum']
        return agg_params
