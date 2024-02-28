import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image



class FedCelebaDataset(Dataset):

    def __init__(self, root_dir='/home/mikonsta/data/celeba/celeba', kwargs={'split': 'test'}):
        '''
        root_dir: path to the image folder
        split_scheme : dict with splitting information
        id: id of the corresponding client
        '''

        self._data_dir = root_dir

        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'meta_data_fed.csv'))

        if kwargs['split'] == 'test':
            self.annotation = 'test'
        else:
            self.annotation = str(kwargs['client_id'])

        self._darray = self._get_darray()


        self.custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                       transforms.Resize((128, 128)),
                                       transforms.ToTensor()])

    def _get_darray(self):
        _x = self._metadata_df[self._metadata_df['Anno'] == self.annotation][['Image', 'Smiling']]
        return np.array(_x)
    
    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img_filename = os.path.join(
            self._data_dir,
            'img_align_celeba',
            self._darray[idx][0])
        x = Image.open(img_filename).convert('RGB')
        y = self._darray[idx][1]
        return x, y
    
    def __len__(self):
        return len(self._darray)

    def __getitem__(self, idx):
        x, y = self.get_input(idx)
        x = self.custom_transform(x)
        return x, y  
