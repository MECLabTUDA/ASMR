import os
import cv2

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split


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

        img = cv2.imread(self.X[idx])
        lbl = self._get_label(self.y[idx])

        return img, lbl