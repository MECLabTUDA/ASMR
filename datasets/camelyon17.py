# Dataloaders

import os
import pandas as pd
import torch
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from PIL import Image

TEST_CENTER = 2


def get_datasets(n_clients, split_scheme, root_dir):
    '''
    get a set of datasets

    n_clients: the number of clients
    split_scheme: dict with splitting information
    root_dir: path to the image folder
    returns: dict of clients with corresponding datasets
    '''
    datasets = {}
    for i in range(n_clients):
        client_dataset = FedCamelyon17Dataset(root_dir, split_scheme, i)
        datasets[i] = client_dataset

    return datasets


class FedCamelyon17Dataset: #(WILDSDataset):

    def __init__(self, root_dir='data', split_scheme='official', id=1, split=None):
        '''
        root_dir: path to the image folder
        split_scheme : dict with splitting information
        id: id of the corresponding client
        '''
        self._data_dir = root_dir
        # self.self_original_resolution
        self.split = split

        # Read metadata
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'})

        self._x_array = self.get_x()
        self._y_array = self.get_y()
        # TODO: Transform in __getitem__

    def get_x(self):
        '''
        get the images from the dataframe
        '''
        x_metadata = self._metadata_df[self._metadata_df['slide'].isin(self.split)]
        x_array = [
            f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            for patient, node, x, y in
            x_metadata.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)]
        return x_array

    def get_y(self):
        '''
        read labels from the metadata dataframe
        '''
        y_array = torch.LongTensor(self._metadata_df[self._metadata_df['slide'].isin(self.split)]['tumor'].values)
        return y_array

    def assign_splits(self):
        '''
        assign TEST data to the metadata dataframe
        assign split scheme to the metadata_df
        '''
        pass

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img_filename = os.path.join(
            self._data_dir,
            self._x_array[idx])
        x = Image.open(img_filename).convert('RGB')
        return x

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self._y_array[idx]
        #metadata = self.metadata_array[idx]
        return x, y #, metadata

