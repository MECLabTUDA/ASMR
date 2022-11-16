from torch.utils.data import DataLoader
from datasets.camelyon17 import FedCamelyon17Dataset
from datasets.camelyon17 import get_datasets


def get_test_loader(root_dir, batch_size, **loader_kwargs):
    return DataLoader(FedCamelyon17Dataset(root_dir),
                      shuffle=False,
                      sampler=None,
                      batch_size=batch_size,
                      **loader_kwargs
                      )


def get_train_loaders(root_dir, batch_size, **loader_kwargs):
    pass

# train_loader = get_train_loader("standard", train_data, batch_size=16)
