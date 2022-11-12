from ..models.get_arch import get_arch
from ..aggregation.aggregations import get_aggregation


def retrieve_server(args):
    pass


class Server:
    def __init__(self, cfg, clients):
        """
        args describing:
        - aggregation Class
        - Model Class
        - set of Clients
        - path to global model
        - root path to local models
        """

        self.aggregation = get_aggregation(cfg['agg_method'])()
        self.model = get_arch(cfg['arch'], False)
        self.path = cfg['global_model_path']
        self.clients = clients

        self.aggregation.get_info()

    def aggregate(self):
        '''
        aggregate the local models to a global model
        '''
        pass

    def run_round(self):
        '''
        triggers one training round with the clients
        '''
        pass

    def _add_client(self):
        '''
        adds a client
        '''
        pass

    def print_information(self):
        pass

    def evaluate(self):
        '''
        evaluates the global model on test data
        '''
        pass
