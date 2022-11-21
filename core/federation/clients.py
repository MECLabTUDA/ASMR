import torch
import os
import shutil

from ..models.get_arch import get_arch
from utils.data_loaders import get_train_loader
from utils.get_trainer import get_trainer

def retrieve_clients(cfg):
    n_clients = cfg['n_clients']
    root_dir = cfg['root_dir']
    batch_size = cfg['batch_size']
    clients = []

    for i in range(n_clients):
        ldr = get_train_loader(root_dir, batch_size, n_clients, i)
        client = Client(cfg, i, ldr)
        clients.append(client)
    return clients


def clean_clients(clients):
    try:
        for client in clients:
            client.clean()
        print("Successfully cleaned up all clients")
    except:
        print("Failed to clean up all clients, manually cleanup might be needed")


class Client:
    def __init__(self, cfg, client_id, ldr):
        """
        Constructor:
        - creates path
        - creates and load global model
        -
        """
        self.model = get_arch(cfg['arch'])
        self.trainer = None
        self.id = client_id
        self.local_model_path = cfg['local_model_root'] + '/' + str(self.id)
        self.global_model_path = cfg['global_model_path']
        self.ldr = ldr
        self.trainer = get_trainer(cfg['trainer'], self.model, self.ldr, cfg['hp_cfg'], self.local_model_path)

    #(trainer, model, ldr, hp_cfg, local_model_path)

        if not os.path.exists(self.local_model_path):
            os.makedirs(self.local_model_path)

    def train(self):
        '''
        Trains Clients model for one Episode
        '''
        self.trainer.train()

    def clean(self):
        try:
            shutil.rmtree(self.local_model_path)
            # logging: successfully cleaned up client store
        except:
            print("could not clean up client: " + str(self.id))
            # logging:

    def _save_model(self):
        self.model.save(self.local_model_path + '/local_model_' + str(self.id) + '.pt')

    def _load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.global_model_path))
            print("Loading global model successfully for Client: " + str(self.id))
            # logging : loading global model was successful
        except:
            print("Failed to load model for client: " + str(self.id))
            # logging: loading model failed for client...

    def update_model(self):
        self._load_model()
