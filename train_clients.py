from utils.read_config import get_configs
from core.federation.server import Server
from core.federation.clients import Client

from core.federation.clients import retrieve_clients, clean_clients
import utils.custom_multiprocess as cm
from torch.multiprocessing import set_start_method, Queue
import numpy as np
import random
import torch

torch.multiprocessing.set_sharing_strategy('file_system')


# Setup Functions
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# q is the client info
def init_process(q, Client, seed):
    set_random_seed(seed)
    global client
    ci = q.get()
    # cfg, i, ldr
    client = Client(ci[0], ci[1], ci[2])


def run_clients(recieved_info):
    try:
        return client.train(recieved_info)
    except KeyboardInterrupt:
        logger.info('exiting')
        return None


# Get the configs
def train_clients(cfg):
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    client_cfg, server_cfg, experiment_cfg = get_configs('configs/' + cfg)
    n_rounds = experiment_cfg['n_rounds']

    # Set up Server and Clients
    clients_info = retrieve_clients(client_cfg)

    server = Server(server_cfg)

    pool = cm.MyPool(processes=client_cfg['n_clients'], initializer=init_process,
                     initargs=(clients_info, Client, experiment_cfg['seed']))

    ##Training of the clients

    # global model = the inital weights = init_model = densenet
    global_weight = server.model.state_dict()

    # initally round 0
    recieved_info = [{'global_weight': global_weight, 'n_round': 0} for x in client_cfg['n_clients']]
    for n_round in range(n_rounds):
        client_outputs = pool.map(run_clients, recieved_info)

        recieved_info = server.operate(client_outputs, n_round)

        # self.aggregate()
        #
        # acc = self.evaluate()
        # add_scalar('Server Test Acc.', acc, global_step=n_round)

    # server.aggregate()
    ##Launching SIA Attack
    # List of state dicts
    # Create net arch
    # Looping through the state dicts of the clients to evaluate the SIA attacks

    # clean_clients(clients)
