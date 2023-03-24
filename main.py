from experiment import experiment
from train_clients import train_clients
import logging
import os
import argparse

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,3,4,5,6,7"

if __name__ == '__main__':
    '''
    Main method
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_path', type=str, default=None, metavar='N',
                        help='path to experiment ./configs/**.yml')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    parser.add_argument('--logfile', type=str, default=None, help='Name of logfile')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logging.basicConfig(filename='logs/' + args.logfile, filemode='w',
                        format='%(asctime) - s%(levelname)s - %(message)s',
                        level=logging.DEBUG)

    logging.info('experiment is starting')

    # Get the configs
    train_clients(args.exp_path)
