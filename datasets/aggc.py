import os

import pandas as pd
from torch.utils.data import Dataset
import tifffile as tiff
import zarr


class FedAggcDataset(Dataset):

    def __init__(self, root_dir='/local/scratch/AGGC', kwargs={'split': 'train'}):

        #  root_path = '/local/scratch/AGGC'

        if kwargs['split'] == 'train':
            self.image_path = os.path.join(root_dir, 'AGGC2022_train/Subset3_Train_image')
            self.annotation_path = os.path.join(root_dir, 'AGGC2022_train/Subset3_Train_annotations_new')
            self.tile_file = os.path.join(root_dir, 'AGGC2022_train/Subset3_Train_tiles/valid_tiles_new.pkl')
        else:
            self.image_path = os.path.join(root_dir, 'AGGC2022_test/Subset3_Test_image')
            self.annotation_path = os.path.join(root_dir, 'AGGC2022_test/Subset3_Test_annotations_new')
            self.tile_file = os.path.join(root_dir, 'AGGC2022_test/Subset3_Test_tiles/cmplt_tiles.pkl')

        self.metadata = pd.read_pickle(self.tile_file)

    def _read_tif_region(self, image, from_x=None, to_x=None, from_y=None, to_y=None):
        '''
        returns a region of an image as numpy array
        '''
        img_tiff = tiff.imread(image, aszarr=True)
        img_zarr = zarr.open(img_tiff, mode="r")

        if isinstance(img_zarr, zarr.hierarchy.Group):
            img_zarr = img_zarr['0']

        if from_x == None:
            return img_zarr

        # return img_zarr[from_y:to_y, from_x:to_x]
        return img_zarr[from_x:to_x, from_y:to_y]

    def _get_item(self, index):
        info = self.metadata.iloc[[index]]
        tile = list(info['tiles'])[0]
        scanner = list(info['scanner'])[0]
        image = list(info['image'])[0]
        image_file = os.path.join(self.image_path, scanner, image)
        mask_file = os.path.join(self.annotation_path, scanner, image[:-5], 'segmentation_mask.tif')

        x = self._read_tif_region(image_file, from_x=tile[0], to_x=tile[1], from_y=tile[2], to_y=tile[3])
        y = self._read_tif_region(mask_file, from_x=tile[0], to_x=tile[1], from_y=tile[2], to_y=tile[3])

        # return np.transpose(x, (3,1,2,0)), np.transose(y, (1,2,0))
        return x, y

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self._get_item(index)
