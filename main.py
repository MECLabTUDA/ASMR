from utils.read_config import read_config
from core.federation.server import Server

if __name__ == '__main__':
    '''
    #Loading the federated setup
    clients, server, cfg = load_federated_config(args)
  
    #in case of sampling
    FedStain.supervise()
  
  
    #Supervisor describes aggregation technique
  
    #Training
    supervisor.supervise(cfg, evaluator)
  
    #Testing
    supervisor.evaluate()
  
    #running method to be executed on the single clients
  
    #clients are managed over single paths
  
    #server model managed over specific path
    '''
    cfg = read_config('configs/camelyon17_base.yml')
    server = Server(cfg, [])
