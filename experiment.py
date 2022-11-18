from utils.read_config import read_config
from core.federation.server import Server
from core.federation.clients import retrieve_clients, clean_clients


def experiment(cfg):
    cfg = read_config('configs/' + cfg)
    clients = retrieve_clients(cfg['client'])
    server = Server(cfg['server'], clients)

    n_rounds = cfg['experiment']['n_rounds']

    for round in range(n_rounds):
        server.run_round()

    server.evaluate()

    clean_clients(clients)
