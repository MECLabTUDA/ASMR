from utils.read_config import read_config
from core.federation.server import Server
from core.federation.clients import retrieve_clients, clean_clients
from utils.data_loaders import get_test_loader
from datasets.camelyon17 import get_datasets

from datasets.camelyon17 import FedCamelyon17Dataset
if __name__ == '__main__':
    '''
    Main method
    '''
    
