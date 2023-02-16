import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import os
import logging
from torch.utils.tensorboard import SummaryWriter

import logging
import sys

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)


class DenseNet121Trainer:
    def __init__(self, model, client_id, ldr, local_model_path, n_local_epochs, device=0):
        '''
        model: model to be trained
        ldr: dataloader
        hp_cfg: configuration of hyperparameters
        '''
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
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        self.tb = SummaryWriter(os.path.join(self.local_model_path, 'log'))

    #TODO: Load model and optimizer
    def train(self, n_round, model, ldr, client_id):
        train_loss = 0
        total = 0
        correct = 0
        self.model = model
        self.ldr = ldr
        self.id = client_id
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)

        self.model.train()
        self.model.to(self.device)

        logger.info('********Training of Client: ' + str(self.id) + '*********')
        epoch_loss = []

        for epoch in range(self.n_local_epochs):
            batch_loss = []

            for batch_index, (inputs, targets) in enumerate(self.ldr, 0):
                # inputs.cuda()

                # targets = torch.FloatTensor(np.array(targets).astype(float)).cuda()

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()

                inputs, targets = Variable(inputs), Variable(targets)
                outputs = self.model(inputs)
                outputs = torch.squeeze(outputs)
                loss = self.criterion(outputs, targets)

                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=5.0)

                self.optimizer.step()

                train_loss += loss.data  # [0]
                total += targets.size(0)

                outputs = outputs.argmax(dim=1)

                correct += outputs.data.eq(targets.data).cpu().sum()
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

        # print(f"Finished training for Client:{self.id}, loss:{loss}, " + str(self.id))
        self.save_local_model(n_round)
        # self.tb.close()
        weights = self.model.cpu().state_dict()
        return weights

    def save_local_model(self, n_round):
        torch.save(self.model.state_dict(), self.local_model_path + str(self.id)
                   + '/local_model_' + str(self.id) + '_round_' + str(n_round) + '.pt')

        torch.save(self.model.state_dict(), self.local_model_path + str(self.id)
                   + '/local_model_' + str(self.id) + '.pt')

        # logger.info("saved local model of Client: " + str(self.id))
