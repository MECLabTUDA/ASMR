from utils.read_config import read_config
from core.federation.server import Server
from core.federation.clients import retrieve_clients, clean_clients

from datasets.camelyon17 import get_datasets
if __name__ == '__main__':
    '''
    Main method
    '''

    d = get_datasets(6, '/Users/mirkokonstantin/tud/master-thesis/project/data')

    for i in range(6):
        d[i].print_information()



