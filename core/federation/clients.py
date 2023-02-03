import torch
import os
import shutil

from ..models.get_arch import get_arch
from utils.data_loaders import get_train_loader
from core.trainers.get_trainer import get_trainer
from torch.multiprocessing import Queue
import logging
import sys

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)

torch.multiprocessing.set_sharing_strategy('file_system')


def retrieve_clients(cfg):
    client_info = Queue()

    n_clients = cfg['n_clients']
    root_dir = cfg['data_root']
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']

    for i in range(n_clients):
        ldr = get_train_loader(root_dir, batch_size, n_clients, i, num_workers=num_workers, pin_memory=True)
        client_info.put((cfg, i, ldr))

        logger.info(f'Client: {i} has {len(ldr.dataset)} samples ')

        # client = Client(cfg, i, ldr)
        logger.debug('created client: ' + str(i))
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
                                                   self.id % torch.cuda.device_count())
        self.dp_scale = cfg['dp_scale']
        self.fl_attack = cfg['fl_attack']
        self.num_samples = len(self.ldr.dataset)
        if not os.path.exists(self.local_model_path):
            os.makedirs(self.local_model_path)

    def train(self, recieved_info):
        '''
        Trains Clients model for one Episode
        '''

        # update the model before training should be abstracted from the server side for multiprocessing?
        self._load_model(recieved_info['global_weight'])

        client_weight = self.trainer.train(recieved_info['n_round'])
        # TODO add gausian noise here (before sending to server?

        if self.fl_attack == 'DP':
            client_weight = add_gaussian_noise(client_weight, self.dp_scale)

        return {'weights': client_weight,
                'num_samples': self.num_samples,
                'n_round': recieved_info['n_round']}

    def clean(self):
        try:
            shutil.rmtree(self.local_model_path)
            logger.debug('successfully cleaned client: ' + str(self.id))
        except:
            logger.error('Unable to clean up client: ' + str(self.id))

    def _save_model(self):
        self.model.save(self.local_model_path + '/local_model_' + str(self.id) + '.pt')

    def _load_model(self, model_weights=None):
        try:
            # to avoid input/output overhead
            if model_weights:
                self.model.load_state_dict(model_weights)
            else:
                self.model.load_state_dict(torch.load(self.global_model_path))
            logger.debug("Loading global model successfully for client: " + str(self.id))
        except:
            logger.error("Failed to load model for client: " + str(self.id))

    def update_model(self):
        self._load_model()
