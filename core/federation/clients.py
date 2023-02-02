import torch
import os
import shutil
import logging

from ..models.get_arch import get_arch
from utils.data_loaders import get_train_loader
from core.trainers.get_trainer import get_trainer

from torch.multiprocessing import  Queue

def retrieve_clients(cfg):
    client_info = Queue()

    n_clients = cfg['n_clients']
    root_dir = cfg['data_root']
    batch_size = cfg['batch_size']
    for i in range(n_clients):
        ldr = get_train_loader(root_dir, batch_size, n_clients, i)
        client_info.put((cfg, i, ldr))
        # client = Client(cfg, i, ldr)
        logging.debug('created client: ' + str(i))
    return client_info


def clean_clients(clients):
    for client in clients:
        client.clean()


class Client:
    def __init__(self, cfg, client_id, ldr):
        """
        Constructor:
        - creates path
        - creates and load global model
        -
        """
        self.model = get_arch(cfg['arch'])
        self.id = client_id
        self.local_model_path = cfg['local_model_root'] + '/' + str(self.id)
        self.global_model_path = cfg['global_model_path']
        self.n_local_epochs = cfg['n_local_epochs']
        self.ldr = ldr
        self.trainer = get_trainer(cfg['trainer'])(self.model, self.id, self.ldr,
                                                   self.local_model_path, self.n_local_epochs,
                                                   i % torch.cuda.device_count())

        if not os.path.exists(self.local_model_path):
            os.makedirs(self.local_model_path)

    def train(self, n_round, global_weights=None):
        '''
        Trains Clients model for one Episode
        '''

        # update the model before training should be abstracted from the server side for multiprocessing?
        self._load_model(global_weights)

        client_weights = self.trainer.train(n_round)
        return client_weights

    def clean(self):
        try:
            shutil.rmtree(self.local_model_path)
            logging.debug('successfully cleaned client: ' + str(self.id))
        except:
            logging.error('Unable to clean up client: ' + str(self.id))

    def _save_model(self):
        self.model.save(self.local_model_path + '/local_model_' + str(self.id) + '.pt')

    def _load_model(self, model_weights=None):
        try:
            # to avoid input/output overhead
            if model_weights:
                self.model.load_state_dict(model_weights)
            else:
                self.model.load_state_dict(torch.load(self.global_model_path))
            logging.debug("Loading global model successfully for client: " + str(self.id))
        except:
            logging.error("Failed to load model for client: " + str(self.id))

    def update_model(self):
        self._load_model()
