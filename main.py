from utils.read_config import read_config
from core.federation.server import Server
from core.federation.clients import retrieve_clients, clean_clients
from utils.data_loaders import get_test_loader
from datasets.camelyon17 import get_datasets

from datasets.camelyon17 import FedCamelyon17Dataset

import logging

if __name__ == '__main__':
    '''
    Main method
    '''
    logging.basicConfig(filename='logs/fed_train.log', filemode='w', format='%(asctime) - s%(levelname)s - %(message)s',
                        level=logging.DEBUG)

    logging.info('experiment is starting')

    # Get the configs
    cfg = read_config('configs/camelyon17_base.yml')

    client_cfg = cfg['client']
    server_cfg = cfg['server']
    exp_cfg    = cfg['experiment']


    #Set up Server and Clients
    logging.debug('Creating Clients')
    clients = retrieve_clients(client_cfg)
    logging.debug('creating server')
    server = Server(server_cfg, clients)

