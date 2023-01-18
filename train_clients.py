from utils.read_config import get_configs
from core.federation.server import Server
from core.federation.clients import retrieve_clients, clean_clients

# Get the configs
def train_clients(cfg):
    client_cfg, server_cfg, experiment_cfg = get_configs('configs/' + cfg)
    n_rounds = experiment_cfg['n_rounds']

    # Set up Server and Clients
    clients = retrieve_clients(client_cfg)
    server = Server(server_cfg, clients)

    ##Training of the clients

    for n_round in range(n_rounds):
        server.run_round(n_round)

    ##Launching SIA Attack
    # List of state dicts
    # Create net arch
    # Looping through the state dicts of the clients to evaluate the SIA attacks

    # clean_clients(clients)