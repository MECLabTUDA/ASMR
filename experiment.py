from utils.read_config import get_configs
from core.federation.server import Server
from core.federation.clients import retrieve_clients, clean_clients


def experiment(cfg, n_rounds):

    # Get the configs

    client_cfg, server_cfg = get_configs('configs/' + cfg)


    #Set up Server and Clients
    clients = retrieve_clients(client_cfg)
    server = Server(server_cfg, clients)
    clean_clients(clients)

