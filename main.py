
from experiment import experiment

import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "4"



if __name__ == '__main__':
    '''
    Main method
    '''
    logging.basicConfig(filename='logs/fed_train.log', filemode='w', format='%(asctime) - s%(levelname)s - %(message)s',
                        level=logging.DEBUG)

    logging.info('experiment is starting')

    # Get the configs
    experiment('camelyon17_base.yml')
