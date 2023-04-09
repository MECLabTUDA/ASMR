import os
import random

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import torchvision.transforms.functional as TF


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


def transform_samples(image, mask, device):
    # Resize
    resize = transforms.Resize(size=(520, 520))
    image = resize(image)
    mask = resize(mask)



    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(512, 512))

    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    # Random gaussian blur
    if random.random() > 0.5:
        image = TF.gaussian_blur(image, kernel_size=5, sigma=(0.5, 2.0))

    # Add random noise
    # if random.random() > 0.5:
    #    image = image + torch.randn(image.size()) * 0.5

    if random.random() > 0.5:
        image = TF.adjust_contrast(image, contrast_factor=0.25)

    if random.random() > 0.5:
        image = TF.adjust_brightness(image, brightness_factor=0.5)

    # if random.random() > 0.5:
    #    image = TF.affine(image, angle=30, translate=(50, 50), scale=1.2, shear=0)

    # Transform to tensor
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    return image, mask


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

        x = Image.fromarray(x.astype('uint8'))
        y = Image.fromarray(y.astype('uint8'))
        return x, y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms

        x, y = self.get_input(idx)
        #x, y = transform_samples(x, y)

        return x, y
