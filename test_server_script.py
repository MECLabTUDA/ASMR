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
import copy
from collections import OrderedDict

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)

# Get the configs

client_cfg, server_cfg, experiment_cfg = get_configs('configs/camelyon17_base.yml')
n_rounds = experiment_cfg['n_rounds']

# Set up Server and Clients
clients_init, clients_info = retrieve_clients(client_cfg)

server = Server(server_cfg, [])

global_weight = copy.deepcopy(server.model.state_dict())
global_weight_2 = server.model.state_dict()

# server.evaluate(global_weight)
zero_state_dict = OrderedDict()
for key, value in global_weight.items():
    zero_state_dict[key] = torch.randn_like(value.float()).float()

logger.info('First evaluation Zeros')
server.evaluate(zero_state_dict)

ones_state_dict = OrderedDict()
for key, value in global_weight.items():
    ones_state_dict[key] = torch.randn_like(value.float()).float()

logger.info('Second evaluation Ones')
server.evaluate(ones_state_dict)


logger.info('Third evaluation no deep')
server.evaluate(global_weight_2)

logger.info('fourth evaluation copy deep')
server.evaluate(global_weight)


