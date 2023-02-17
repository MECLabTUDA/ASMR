from torch.utils.data import DataLoader
from datasets.camelyon17 import FedCamelyon17Dataset


kwargs = {'n_clients': 2, 'test_split': False, 'client_id': 1}

root_dir = "/local/scratch/camelyon17/camelyon17_v1.0"

loader_kw

ldr = DataLoader(FedCamelyon17Dataset(root_dir, kwargs),
                                        shuffle=True,
                                        sampler=None,
                                        batch_size=batch_size,
                                        drop_last=True,
                                        **loader_kwargs
                                        )