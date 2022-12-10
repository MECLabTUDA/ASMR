from utils.read_config import read_config
from core.federation.server import Server
from core.federation.clients import retrieve_clients, clean_clients
from experiment import experiment

import logging

if __name__ == '__main__':
    '''
    Main method
    '''
    logging.basicConfig(filename='logs/fed_train.log', filemode='w', format='%(asctime) - s%(levelname)s - %(message)s',
                        level=logging.DEBUG)

    logging.info('experiment is starting')

    # Get the configs
    experiment('camelyon17_base.yml', 1)
