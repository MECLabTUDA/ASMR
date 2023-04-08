import math
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.optim import Adam
from tqdm import tqdm

from core.attacks.ana import add_gaussian_noise
from core.attacks.sfa import flip_signs

import logging
import sys

import segmentation_models_pytorch as smp
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter


from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)


class GlasUnetTrainer:
    def __init__(self, model, client_id, ldr, local_model_path, n_local_epochs, device=0):
        self.id = client_id
        self.device = device
        self.model = model
        self.local_model_path = local_model_path
        self.n_local_epochs = n_local_epochs
        self.ldr = ldr
        self.batch_size = 1
        self.lr = 1e-3

        tr_transforms = self.get_train_transform()

        self.train_gen = MultiThreadedAugmenter(self.ldr, tr_transforms, num_processes=4,
                                           num_cached_per_queue=2,
                                           seeds=None, pin_memory=False)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=self.lr)
        self.tb = SummaryWriter(os.path.join(self.local_model_path, 'log'))

        self.dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.xent = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1)

    def min_max_norm(self, img, axis=(1, 2)):
        """
        Channel-wise Min max normalization for
        images with input [batch size, slices, width, channel]
        @param img: Input image of 4D array
        @return: Min max norm of the image per channel
        """
        inp_shape = img.shape
        img_min = np.broadcast_to(img.min(axis=axis, keepdims=True), inp_shape)
        img_max = np.broadcast_to(img.max(axis=axis, keepdims=True), inp_shape)
        x = (img - img_min) / (img_max - img_min + float(1e-18))
        return x

    def custom_loss(self, pred, target):
        xent_l = self.xent(pred, target)
        dice_l = self.dice_loss(pred, target)
        loss = xent_l + dice_l
        return loss, xent_l, dice_l

    def get_train_transform(self, patch_size=(512,512), prob=0.5):
        # We now create a list of transforms.
        # These are not necessarily the best transforms to use for BraTS, this is just
        # to showcase some things
        tr_transforms = []

        # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
        # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
        # shape and do not transform spatially, so no border artifacts will be introduced
        # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
        # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
        # of samples will be augmented, the rest will just be cropped
        tr_transforms.append(
            SpatialTransform(
                patch_size,
                [i // 2 for i in patch_size],
                do_elastic_deform=True,
                alpha=(0., 300.),
                sigma=(20., 40.),
                do_rotation=True,
                angle_x=(-np.pi / 15., np.pi / 15.),
                angle_y=(-np.pi / 15., np.pi / 15.),
                angle_z=(0., 0.),
                do_scale=True,
                scale=(1 / 1.15, 1.15),
                random_crop=False,
                border_mode_data='constant',
                border_cval_data=0,
                order_data=3,
                p_el_per_sample=prob, p_rot_per_sample=prob, p_scale_per_sample=prob
            )
        )

        # now we mirror along the y-axis
        tr_transforms.append(MirrorTransform(axes=(1,)))

        # brightness transform
        tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=prob))

        # Gaussian Noise
        tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.5), p_per_sample=prob))

        # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
        # thus make the model more robust to it
        tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 2.0), different_sigma_per_channel=True,
                                                   p_per_channel=prob, p_per_sample=prob))
        tr_transforms.append(ContrastAugmentationTransform(contrast_range=(0.75, 1.25), p_per_sample=prob))
        # now we compose these transforms together
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    def step(self):
        #TODO: New Dataloader
        #num_batches = math.ceil(len(ds_dict['train_ds']['img_npy']) / self.batch_size)
        self.model.train()
        batch_xent_l = []
        batch_dice_l = []
        batch_loss = []

        #for i in tqdm(range(num_batches)):
        for imgs, segs in tqdm(self.train_gen):

            # normalization
            imgs = self.min_max_norm(imgs)
            # binarisation
            segs = np.where(segs > 0., 1.0, 0.).astype('float32')
            segs = np.expand_dims(segs[:, 0, :, :], 1)
            imgs, segs = torch.from_numpy(imgs).to(self.device), torch.from_numpy(segs).to(self.device)
            # Compute loss
            pred = self.model(imgs)
            loss, xent_l, dice_l = self.custom_loss(pred, segs)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # batch losses
            batch_xent_l.append(xent_l)
            batch_dice_l.append(dice_l)
            batch_loss.append(loss)
            # apply sigmoid to masking
        segs = nn.Sigmoid()(segs)
        # taking the average along the batch
        loss = torch.mean(torch.as_tensor(batch_loss)).item()
        avg_xent_l = torch.mean(torch.as_tensor(batch_xent_l)).item()
        avg_dice_l = torch.mean(torch.as_tensor(batch_dice_l)).item()

        return {'loss': loss, 'xent_l': avg_xent_l, 'dice_l': avg_dice_l,
                'imgs': imgs.cpu().detach().numpy(),
                'segs': segs.cpu().detach().numpy(),
                'pred': pred.cpu().detach().numpy()}

    def train(self, n_round, model, ldr, client_id, fl_attack=None, dp_scale=None):

        current_total_loss = 1000
        current_dice_score = 0

        self.model = model
        self.ldr = ldr
        self.id = client_id

        self.optimizer = Adam(model.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=20, factor=0.1)

        self.model.train()
        self.model.to(self.device)

        if fl_attack is None:
            logger.info('********Training of Client: ' + str(self.id) + '*********')
        else:
            logger.info(f'******** Malicious ({fl_attack}) Training of Client: ' + str(self.id) + '*********')

        epoch_loss = []

        for epoch in range(self.n_local_epochs):
            train_output = self.step()
            #test_output = test(model)
            scheduler.step(train_output['loss'])
            '''
            self.scheduler.step(test_output['loss'])

            if epoch % 10 == 0:
                # threshold sigmoid output with 0.5
                pred_thr = np.where(test_output['pred'] > 0.5, 1.0, 0.0)
                # sample a dataset from the batch for visualization purpose
                imgs = [test_output['imgs'][0, 0, :, :], test_output['segs'][0, 0, :, :], pred_thr[0, 0, :, :]]
                captions = ['Gland Image', 'Masking', 'Prediction']
                # fig = utils.plot_comparison(imgs, captions, plot=False, n_col=len(imgs),
                #                      figsize=(12, 12), cmap='gray')
            '''
        # self.tb.close()
        weights = self.model.cpu().state_dict()
        if fl_attack == 'ana':
            weights = add_gaussian_noise(weights, dp_scale)
        elif fl_attack == 'sfa':
            weights = flip_signs(weights, dp_scale)

        self._save_local_model(n_round, weights)
        return weights

    def _save_local_model(self, n_round, state_dict):
        torch.save(state_dict, self.local_model_path + str(self.id)
                   + '/local_model_' + str(self.id) + '_round_' + str(n_round) + '.pt')

        torch.save(state_dict, self.local_model_path + str(self.id)
                   + '/local_model_' + str(self.id) + '.pt')
