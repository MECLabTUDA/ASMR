from torch.utils.data import DataLoader
from datasets.camelyon17 import FedCamelyon17Dataset
from datasets.camelyon17 import get_datasets


def get_test_loader(root_dir, batch_size=4, **loader_kwargs):
    return DataLoader(FedCamelyon17Dataset(root_dir),
                      shuffle=False,
                      sampler=None,
                      batch_size=batch_size,
                      **loader_kwargs
                      )


def get_train_loaders(root_dir, batch_size, clients, **loader_kwargs):
    loaders = {}
    kwargs = {'n_clients': len(clients), 'test_split': False}
    for client in clients:
        client_id = client.id
        kwargs['client_id'] = client_id
        loaders[client.id] = DataLoader(FedCamelyon17Dataset(root_dir, kwargs),
                                        shuffle=False,
                                        sampler=None,
                                        batch_size=batch_size,
                                        **loader_kwargs
                                        )
    return loaders


def get_train_loader(root_dir, batch_size, n_clients, client_id, **loader_kwargs):
    kwargs = {'n_clients': n_clients, 'test_split': False, 'client_id': client_id}
    ldr = DataLoader(FedCamelyon17Dataset(root_dir, kwargs),
                     shuffle=False,
                     sampler=None,
                     batch_size=batch_size,
                     **loader_kwargs
                     )
    return ldr
