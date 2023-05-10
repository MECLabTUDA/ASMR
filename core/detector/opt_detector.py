class Optimal_detector:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.clients_info = None


    def fit(self, clients_info):
        self.clients_info = clients_info

    def detect(self):

        # Get the IDs of the benign clients
        benign_clients = []

        for client in self.clients_info:
            if client['attack']:
                benign_clients.append(client['id'])

        # Assign clients to
        mal_clients = []
        ben_clients = {}

        for client in self.clients_info:

            if self.clients_info[client]['id'] in benign_clients:
                client_id = self.clients_info[client]['id']
                ben_clients[client_id] = self.clients_info[client]
            else:
                mal_clients.append(self.clients_info[client]['id'])

        return ben_clients, mal_clients
