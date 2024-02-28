import os
import cv2
import random

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from froodo import *



class FedCrcDataset(Dataset):

    def __init__(self, root_dir='/local/scratch/NCT-CRC-HE/NCT-CRC-HE-100K', kwargs={'split': 'test'}):

        self.root_dir = root_dir

        if kwargs['split'] == 'train':
            self.n_clients = kwargs['n_clients']
            self.client_id = kwargs['client_id']
            self.annotation = 'train'
        else:
            self.annotation = 'test'

        self.X, self.y = self._read_img_names(root_dir)

        self.artifacts_list = self._artifact_list()

        self.artifacts = False

    def _artifact_list(self):
        
        darkspots = DarkSpotsAugmentation(sample_intervals=[(3, 5)],scale=2,keep_ignorred=True)
        fatspots  = FatAugmentation(sample_intervals=[(1., 5)],scale=2, keep_ignorred=True)
        squamous  = SquamousAugmentation(sample_intervals=[(2, 3)],scale=2, keep_ignorred=True)
        thread    = ThreadAugmentation(sample_intervals=[(2, 4)],scale=2, keep_ignorred=True)
        blood     = BloodCellAugmentation(sample_intervals=[(1, 25)],scale=3,scale_sample_intervals=[(1.0, 1.02)])
        blood.scale = 0.1
        bubble    = BubbleAugmentation(base_augmentation=transforms.GaussianBlur(kernel_size=(9, 9),sigma=10))
        bubble.overlay_h = 700
        bubble.overlay_w = 700

        artifact_list = [darkspots, fatspots, squamous, thread, blood, bubble]

        return artifact_list

    
    def set_artifacts(self, artifacts):
        self.artifacts = artifacts


    def _read_img_names(self, root_dir):
        imgs = []
        X = []
        y = []
        for root, _, filenames in os.walk(root_dir):
            for filename in filenames:
                img = os.path.join(root, filename)
                lbl = root.split('/')[-1]
                # imgs.append({'img': img, 'label': lbl})
                X.append(img)
                y.append(lbl)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if self.annotation == 'test':
            return X_test, y_test
        else:
            samples_per_client = len(X_train) // self.n_clients
            first_sample = self.client_id * samples_per_client
            return X_train[first_sample:first_sample + samples_per_client], y_train[
                                                                            first_sample:first_sample + samples_per_client]

    def __len__(self):
        return len(self.y)

    def _get_label(self, lbl):
        lbl_map = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}
        return lbl_map[lbl]

    def __getitem__(self, idx):
        
        img = torch.from_numpy(cv2.imread(self.X[idx]))
        
        if self.artifacts:
            art = random.choice(self.artifacts_list)
            img = Sample((img / 255.).permute(2,0,1))
            img = art(img)
            img = img.image.permute(1,2,0)

        lbl = self._get_label(self.y[idx])

        return img, lbl
