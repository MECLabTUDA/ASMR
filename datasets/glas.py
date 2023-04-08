import os

import pandas as pd
import torch
from PIL.Image import Image
from torchvision.transforms import transforms
import numpy as np


def get_datasets(n_clients, root_dir):
    '''
    get a set of datasets

    n_clients: the number of clients
    split_scheme: dict with splitting information
    root_dir: path to the image folder
    returns: dict of clients with corresponding datasets
    '''
    datasets = {}
    kwargs = {'n_clients': n_clients, 'test_split': False}
    for i in range(n_clients):
        kwargs['client_id'] = i
        client_dataset = FedGlasDataset(root_dir, kwargs)
        datasets[i] = client_dataset

    return datasets


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class FedGlasDataset:

    def __init__(self, root_dir='/local/scratch/glas', kwargs={'split': 'test'}):
        '''
        root_dir: path to the image folder
        split_scheme : dict with splitting information
        id: id of the corresponding client
        '''

        self._data_dir = root_dir

        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'fedmeta.csv'),
            index_col=0,
            dtype={'patient': 'str'})

        if kwargs['split'] == 'test':
            self.annotation = 'test'
        else:
            self.annotation = str(kwargs['client_id'])

        # get the split of the data
        self.samples = self.get_data()
        # TODO: Transform in __getitem__

        self.transform = self.get_transform()
        self.prob = 0.5

    def get_transform(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5),
            AddGaussianNoise(mean=0, std=0.5),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0)),
            transforms.ToTensor()
        ])
        return transform

    def get_data(self):
        '''
        get the images from the dataframe
        '''
        x_metadata = self._metadata_df[self._metadata_df['client_id'] == str(self.annotation)]
        data = [
            (img, anno)
            for img, anno, x, y in
            x_metadata.loc[:, ['img_npy', 'anno_npy', 'image_height', 'image_width']].itertuples(index=False,
                                                                                                 name=None)]
        return data

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        (img_filename, anno_filename) = self.samples[idx]
        # x = Image.open(os.path.join(self._data_dir, img_filename)).convert('RGB')
        # y = Image.open(os.path.join(self._data_dir, anno_filename))

        x = np.load(os.path.join(self._data_dir, img_filename))
        y = np.load(os.path.join(self._data_dir, anno_filename))
        return x, y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms

        transform = transforms.RandomApply([self.transform], p=self.prob)
        x, y = transform(self.get_input(idx))
        '''
        trans = transforms.ToTensor()
        x = trans(x)
        y = trans(y)
        '''
        return x, y
