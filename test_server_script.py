from utils.read_config import get_configs
from core.federation.server import Server
from core.federation.clients import Client

from core.federation.clients import retrieve_clients, clean_clients
import utils.custom_multiprocess as cm
from torch.multiprocessing import set_start_method, Queue
import numpy as np
import random
import torch
import logging
import sys

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)

# Get the configs

client_cfg, server_cfg, experiment_cfg = get_configs('configs/camelyon17_base.yml')
n_rounds = experiment_cfg['n_rounds']

# Set up Server and Clients
clients_init, clients_info = retrieve_clients(client_cfg)

server = Server(server_cfg, [])

global_weight = server.model.state_dict()

# server.evaluate(global_weight)

logger.info('First evaluation fiinished')
server.evaluate(torch.zeros_like(global_weight))
logger.info('zeros evaluation fiinished')
server.evaluate(torch.ones_like(global_weight))
logger.info('ones evaluation fiinished')


