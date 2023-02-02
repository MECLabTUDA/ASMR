
from experiment import experiment
from train_clients import train_clients
import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4"

if __name__ == '__main__':
    '''
    Main method
    '''
    logging.basicConfig(filename='logs/fed_train.log', filemode='w', format='%(asctime) - s%(levelname)s - %(message)s',
                        level=logging.DEBUG)

    logging.info('experiment is starting')

    # Get the configs
    #experiment('camelyon17_base.yml')
    train_clients('camelyon17_base.yml')
