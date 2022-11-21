from utils.read_config import read_config
from core.federation.server import Server
from core.federation.clients import retrieve_clients, clean_clients


def experiment(cfg):

    # Get the configs
    cfg = read_config('configs/' + cfg)

    client_cfg = cfg['client']
    server_cfg = cfg['server']
    exp_cfg    = cfg['experiment']


    #Set up Server and Clients
    clients = retrieve_clients(client_cfg)
    server = Server(server_cfg, clients)


    n_rounds = exp_cfg['n_rounds']

    for round in range(n_rounds):
        server.run_round()



    server.evaluate()

    clean_clients(clients)
