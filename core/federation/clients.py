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
    dataset = cfg['dataset']

    clients_info = {}
    for i in range(n_clients):
        ldr = get_train_loader(root_dir, batch_size, n_clients, i, dataset, num_workers=num_workers, pin_memory=True)
        client_init.put((cfg, i, ldr))

        malicious = False
        status = False
        if i in active_clients:
            status = True
        if i in malicious_clients:
            malicious = True

        logger.info(f'Client: {i} has {len(ldr.dataset)} samples | Active: {status} | Malicious: {malicious}')

        # clients info for first round

        clients_info[i] = {'weights': None,
                           'num_samples': len(ldr.dataset),
                           'n_round': 0,
                           'active': status,
                           'malicious': malicious,
                           'ldr': ldr,
                           'detected': False}

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
        self.mal_clients = cfg['mal_clients']
        if self.id in cfg['mal_clients']:
            self.malicious = True
        else:
            self.malicious = False
        self.local_model_path = cfg['local_model_root'] + '/'
        self.global_model_path = cfg['global_model_path']
        self.n_local_epochs = cfg['n_local_epochs']
        self.ldr = ldr
        self.trainer = get_trainer(cfg['trainer'])(self.model, self.id, self.ldr,
                                                   self.local_model_path, self.n_local_epochs,
                                                   self.id % torch.cuda.device_count())
        self.dp_scale = cfg['dp_scale']
        
        self.fl_attacks = cfg['fl_attack']

        if len(self.fl_attacks) > 0:
            self.fl_attack = random.choice(cfg['fl_attack'])
        else:
            self.fl_attack = None


        self.attack_freq = cfg['attack_prob']
        self.num_samples = len(self.ldr.dataset)
        if not os.path.exists(self.local_model_path + str(client_id)):
            os.makedirs(self.local_model_path + str(client_id))

    def attack(self, prob):
        return random.random() < prob

    def _update_client(self, receive_info):
        self.ldr = receive_info['ldr']
        self.id = receive_info['id']

        if self.id in self.mal_clients:
            self.malicious = True
        else:
            self.malicious = False

        self.num_samples = len(self.ldr.dataset)
        

        if len(self.fl_attacks) > 0:
            self.fl_attack = random.choice(self.fl_attacks)
        else:
            self.fl_attack = None
            

    def train(self, recieved_info):
        '''
        Trains Clients model for one Episode
        '''


        self._update_client(recieved_info)
        

        attack = False

        if self.id not in recieved_info['active_clients']:
            return {'id': self.id,
                    'weights': None,
                    'num_samples': self.num_samples,
                    'n_round': recieved_info['n_round'],
                    'active': False,
                    'malicious': self.malicious,
                    'ldr': self.ldr,
                    'detected': False,
                    'attack': attack}
        else:
            # update the model before training should be abstracted from the server side for multiprocessing?
            

            self._load_model(recieved_info['global_weight'])
            
            
            


            p = self.attack(self.attack_freq)
            if self.malicious and self.attack(self.attack_freq):
                logger.info(f'Client: {self.id} is comitting a malicious udpate')
                attack = True
                
                
                if self.fl_attack == 'ana':
                    client_weight = self.trainer.train(recieved_info['n_round'], self.model, self.ldr, self.id,
                                                       self.fl_attack, self.dp_scale)
                elif self.fl_attack == 'sfa':
                    client_weight = self.trainer.train(recieved_info['n_round'], self.model, self.ldr, self.id,
                                                       self.fl_attack)

                elif self.fl_attack == 'artifacts':
                    client_weight = self.trainer.train(recieved_info['n_round'], self.model, self.ldr, self.id,
                                                       self.fl_attack)
                
            else:
                client_weight = self.trainer.train(recieved_info['n_round'], self.model, self.ldr, self.id)

        return {'id': self.id,
                'weights': client_weight,
                'num_samples': self.num_samples,
                'n_round': recieved_info['n_round'],
                'active': True,
                'malicious': self.malicious,
                'ldr': self.ldr,
                'detected': False,
                'attack': attack}

    def clean(self):
        try:
            shutil.rmtree(self.local_model_path)
            logger.debug('successfully cleaned client: ' + str(self.id))
        except:
            logger.error('Unable to clean up client: ' + str(self.id))

    def _save_model(self):
        self.model.save(self.local_model_path + str(self.id) + '/local_model_' + str(self.id) + '.pt')

    def _load_model(self, model_weights=None):
        
        try:
            # to avoid input/output overhead
            if model_weights:
                

                model_state_dict_keys = set(self.model.state_dict().keys())
                loaded_state_dict_keys = set(model_weights.keys())

                # Check if the keys match
                self.model.load_state_dict(model_weights)
            else:
                self.model.load_state_dict(torch.load(self.global_model_path))
            logger.debug("Loading global model successfully for client: " + str(self.id))
        except:
            logger.error("Failed to load model for client: " + str(self.id))

    def update_model(self):
        self._load_model()
