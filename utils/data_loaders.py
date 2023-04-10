from torch.utils.data import DataLoader
from datasets.camelyon17 import FedCamelyon17Dataset
from datasets.camelyon17 import get_datasets
from datasets.crc import FedCrcDataset
from datasets.glas import FedGlasDataset


def get_dataset(dataset):
    if dataset == 'camelyon17':
        return FedCamelyon17Dataset
    elif dataset == 'glas':
        return FedGlasDataset
    elif dataset == 'crc':
        return FedCrcDataset


def get_test_loader(root_dir, batch_size, dataset, **loader_kwargs):
    FedDataset = get_dataset(dataset)
    return DataLoader(FedDataset(root_dir),
                      shuffle=False,
                      sampler=None,
                      batch_size=batch_size,
                      drop_last=True,
                      **loader_kwargs
                      )


def get_train_loaders(root_dir, batch_size, clients, dataset, **loader_kwargs):
    loaders = {}
    FedDataset = get_dataset(dataset)
    kwargs = {'n_clients': len(clients), 'test_split': False}
    for client in clients:
        client_id = client.id
        kwargs['client_id'] = client_id
        loaders[client.id] = DataLoader(FedDataset(root_dir, kwargs),
                                        shuffle=True,
                                        sampler=None,
                                        batch_size=batch_size,
                                        drop_last=True,
                                        **loader_kwargs
                                        )
    return loaders


def get_train_loader(root_dir, batch_size, n_clients, client_id, dataset, **loader_kwargs):
    kwargs = {'n_clients': n_clients, 'split': 'client', 'client_id': client_id}
    FedDataset = get_dataset(dataset)
    ldr = DataLoader(FedDataset(root_dir, kwargs),
                     shuffle=True,
                     sampler=None,
                     batch_size=batch_size,
                     drop_last=True,
                     **loader_kwargs
                     )
    return ldr
