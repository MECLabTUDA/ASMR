import torch
import os
import shutil

from ..attacks.ana import add_gaussian_noise
from ..models.get_arch import get_arch
from utils.data_loaders import get_train_loader
from core.trainers.get_trainer import get_trainer
from torch.multiprocessing import Queue
import logging
import sys
import random

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)

torch.multiprocessing.set_sharing_strategy('file_system')


def retrieve_clients(cfg):
    client_init = Queue()

    n_clients = cfg['n_clients']
    root_dir = cfg['data_root']
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    active_clients = cfg['starting_clients']
    malicious_clients = cfg['mal_clients']

    clients_info = []
    for i in range(n_clients):
        ldr = get_train_loader(root_dir, batch_size, n_clients, i, num_workers=num_workers, pin_memory=True)
        client_init.put((cfg, i, ldr))

        malicious = False
        status = False
        if i in active_clients:
            status = True
        if i in malicious_clients:
            malicious = True

        logger.info(f'Client: {i} has {len(ldr.dataset)} samples | Active: {status} | Malicious: {malicious}')

        # clients info for first round
        clients_info.append({'weights': None,
                             'num_samples': len(ldr.dataset),
                             'n_round': 0,
                             'active': status,
                             'malicious': malicious})

        # client = Client(cfg, i, ldr)
        logger.debug('created client: ' + str(i))

    return client_init, clients_info


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
        if self.id in cfg['mal_clients']:
            self.malicious = True
        else:
            self.malicious = False
        self.local_model_path = cfg['local_model_root'] + '/' + str(self.id)
        self.global_model_path = cfg['global_model_path']
        self.n_local_epochs = cfg['n_local_epochs']
        self.ldr = ldr
        self.trainer = get_trainer(cfg['trainer'])(self.model, self.id, self.ldr,
                                                   self.local_model_path, self.n_local_epochs,
                                                   self.id % torch.cuda.device_count())
        self.dp_scale = cfg['dp_scale']
        self.fl_attack = cfg['fl_attack']
        self.attack_freq = cfg['attack_prob']
        self.num_samples = len(self.ldr.dataset)
        if not os.path.exists(self.local_model_path):
            os.makedirs(self.local_model_path)

    def attack(self, prob):
        return random.random() < prob

    def train(self, recieved_info):
        '''
        Trains Clients model for one Episode
        '''
        if not recieved_info['active']:
            return {'weights': None,
                    'num_samples': self.num_samples,
                    'n_round': recieved_info['n_round']}
        else:
   
            # update the model before training should be abstracted from the server side for multiprocessing?
            self._load_model(recieved_info['global_weight'])

            client_weight = self.trainer.train(recieved_info['n_round'])

            if self.attack(self.attack_freq):
                logger.info(f'Client: {self.id} is commiting a malicious update in round {recieved_info["n_round"]}')
                if self.fl_attack == 'ana':
                    client_weight = add_gaussian_noise(client_weight, self.dp_scale)
                elif self.fl_attack == '':
                    client_weight = None

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
