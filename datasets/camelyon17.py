# Dataloaders

import os
import pandas as pd
import torch
from PIL import Image


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
        client_dataset = FedCamelyon17Dataset(root_dir, kwargs)
        datasets[i] = client_dataset

    return datasets


class FedCamelyon17Dataset:

    def __init__(self, root_dir='data', kwargs={'test_split': True}):
        '''
        root_dir: path to the image folder
        split_scheme : dict with splitting information
        id: id of the corresponding client
        '''

        self._data_dir = root_dir

        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'})

        if kwargs['test_split'] == True:
            self.assign_splits()
        else:
            self._n_clients = kwargs['n_clients']
            self.client_id = kwargs['client_id']
            self.assign_splits(False)

        # get the split of the data
        self._x_array = self.get_x()
        self._y_array = self.get_y()
        # TODO: Transform in __getitem__

    def print_information(self):
        print('client_id: ' + str(self.client_id))
        print(self.__len__())


    def get_x(self, test=True):

        if test:
            annotation = 'Test'
        else:
            annotation = 'Client'
        '''
        get the images from the dataframe
        '''
        x_metadata = self._metadata_df[self._metadata_df['split'] == annotation]
        x_array = [
            f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            for patient, node, x, y in
            x_metadata.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)]
        return x_array

    def get_y(self):
        '''
        read labels from the metadata dataframe
        '''
        y_array = torch.LongTensor(self._metadata_df[self._metadata_df['split'] == 'Client']['tumor'].values)
        return y_array

    def get_split(self):
        '''
        get corresponding slide id's of this client
        '''
        clients_per_hospital = int(self._n_clients / 4) + 1
        patients_per_client = int(10 / clients_per_hospital)
        hospital = self.client_id // clients_per_hospital

        if self.client_id == 0:
            order_of_client_in_hospital = 0
        else:
            order_of_client_in_hospital = (self.client_id % clients_per_hospital)
        if self.client_id == 0:
            patients_of_client = list(range(0, patients_per_client))
        else:
            patient_start_id = (hospital) * 10 + order_of_client_in_hospital * patients_per_client
            patients_of_client = list(range(patient_start_id, patient_start_id + patients_per_client))

        return patients_of_client

    def assign_splits(self, test=True):
        '''
        labels the metadata, which patients to be selected
        '''
        if test:
            split = self._metadata_df[self._metadata_df['center'] == 4]['slide'].unique()
            annotation = 'Test'
        else:
            split = self.get_split()
            annotation = 'Client'

        client_mask = (self._metadata_df['slide'].isin(split))
        self._metadata_df.loc[client_mask, 'split'] = annotation

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
        return len(self._y_array)

    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self._y_array[idx]
        # metadata = self.metadata_array[idx]
        return x, y  # , metadata
