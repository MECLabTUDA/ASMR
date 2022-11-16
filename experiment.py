from utils.read_config import read_config
from core.federation.server import Server
from core.federation.clients import retrieve_clients, clean_clients


def experiment(cfg):
    cfg = read_config('configs/' + cfg)
    server = Server(cfg)
    clients = retrieve_clients(cfg)

    server.init_clients(clients)

    clean_clients(clients)
