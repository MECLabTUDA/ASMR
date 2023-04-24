import logging
import os
import sys

import torch.nn as nn
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from core.attacks.ana import add_gaussian_noise
from core.attacks.sfa import flip_signs

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)


class ResnetTrainer:
    def __init__(self, model, client_id, ldr, local_model_path, n_local_epochs, device=0):
        self.lr = 0.0003
        self.momentum = 0.9
        self.model = model
        self.ldr = ldr
        self.id = client_id
        self.local_model_path = local_model_path
        self.n_local_epochs = n_local_epochs
        self.device = device
        self.tb = SummaryWriter(os.path.join(self.local_model_path, 'log'))

    def train(self, n_round, model, ldr, client_id, fl_attack=None, dp_scale=None):
        self.model = model
        self.ldr = ldr
        self.id = client_id
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        #self.model.train()
        #self.model.to(self.device)

        model.train()
        model.to(self.device)

        # Prepare

        train_loss = 0
        total = 0
        correct = 0

        epoch_loss = []

        for epoch in range(self.n_local_epochs):
            batch_loss = []
            for img, label in ldr:
                img, label = img.type(torch.FloatTensor).to(self.device).permute(0, 3, 1, 2), label.to(self.device)
                optimizer.zero_grad()

                pred = model(img)

                loss = F.cross_entropy(pred, label)

                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=5.0)
                optimizer.step()

                train_loss += loss.data
                total += label.size(0)

                outputs = pred.argmax(dim=1)

                correct += outputs.data.eq(label.data).cpu().sum()
                batch_loss.append(loss.item())

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logger.info(
                    '(client {}, Round: {}, Local Training Epoch: {} \t Loss: {:.6f} \t correct: {}, train_acc: {:.3f}'.format(
                        self.id, n_round, epoch,
                        sum(
                            epoch_loss) / len(
                            epoch_loss), correct, 100. * correct / total))

            self.tb.add_scalar("Client:" + str(self.id) + "/Loss", sum(epoch_loss) / len(epoch_loss), n_round)
            self.tb.add_scalar("Client:" + str(self.id) + "/Correct", correct, n_round)

            weights = self.model.cpu().state_dict()
            if fl_attack == 'ana':
                weights = add_gaussian_noise(weights, dp_scale)
            elif fl_attack == 'sfa':
                weights = flip_signs(weights)

            self._save_local_model(n_round, weights)
            return weights

    def _save_local_model(self, n_round, state_dict):
        torch.save(state_dict, self.local_model_path + str(self.id)
                   + '/local_model_' + str(self.id) + '_round_' + str(n_round) + '.pt')

        torch.save(state_dict, self.local_model_path + str(self.id)
                   + '/local_model_' + str(self.id) + '.pt')


