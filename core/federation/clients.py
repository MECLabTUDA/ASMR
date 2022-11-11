def retrieve_clients(args):
  '''
  args defining:
  - Number of Clients
  - Path to store the Models
  - The Model that should be trained
  - Path to the data
  '''
  pass

class Client():
  def __init__(self, args):
    '''
    gets following parameters:
    - Model Class
    - Trainer Class
    - ID
    - root_path: root path to the clients local model
    - model_path: path to the weights of the local model

    Constructor:
    - creates path
    - creates and load global model
    -

    '''
    pass

  def train(self):
    '''
    Trains Clients model for one Episode
    '''
    pass

  def _save_model(self):
    '''
    saves state dict to clients path
    '''
    pass

  def _load_model(self):
    '''
    loads the global model to the client
    '''
    pass
