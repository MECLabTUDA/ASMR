import os

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from core.attacks.ana import add_gaussian_noise
from core.attacks.sfa import flig_signs


class Fcn8Trainer:
    def __init__(self, model, client_id, ldr, local_model_path, n_local_epochs, device=0):
        self.id = client_id
        self.device = device
        self.model = model
        self.local_model_path = local_model_path
        self.n_local_epochs = n_local_epochs
        self.ldr = ldr
        self.batch_size = 8
        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.tb = SummaryWriter(os.path.join(self.local_model_path, 'log'))

    def train(self, n_round, model, ldr, client_id, fl_attack=None, dp_scale=None):

        model.train()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        for epoch in self.n_local_epochs:
            running_loss = 0.0
            for i, (data, labels) in enumerate(self.ldr):
                inputs, labels = data.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                loss.backwards()

                optimizer.step()

                running_loss += loss.item()

                ## Could print running loss
        weights = model.cpu().state_dict()

        if fl_attack == 'ana':
            weights = add_gaussian_noise(weights, dp_scale)
        elif fl_attack == 'sfa':
            weights = flig_signs(weights, dp_scale)

        self._save_local_model(n_round, weights)
        return weights

    def _save_local_model(self, n_round, state_dict):
        torch.save(state_dict, self.local_model_path + str(self.id)
                   + '/local_model_' + str(self.id) + '_round_' + str(n_round) + '.pt')

        torch.save(state_dict, self.local_model_path + str(self.id)
                   + '/local_model_' + str(self.id) + '.pt')

