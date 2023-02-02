import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import os

from torch.utils.tensorboard import SummaryWriter


class DenseNet121Trainer:
    def __init__(self, model, client_id, ldr, local_model_path, n_local_epochs):
        '''
        model: model to be trained
        ldr: dataloader
        hp_cfg: configuration of hyperparameters
        '''
        self.id = client_id
        self.model = model
        self.local_model_path = local_model_path
        self.n_local_epochs = n_local_epochs
        self.ldr = ldr
        self.batch_size = 8
        self.lr = 0.005
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)

        self.tb = SummaryWriter(os.path.join(self.local_model_path, 'log'))

    def train(self, n_round):
        train_loss = 0
        total = 0
        correct = 0

        self.model.train()
        self.model.cuda()

        print('********Training of Client: ' + str(self.id) + '*********')
        for epoch in tqdm(range(self.n_local_epochs)):
            for batch_index, (inputs, targets) in enumerate(tqdm(self.ldr), 0):
                # inputs.cuda()
                # targets = torch.FloatTensor(np.array(targets).astype(float)).cuda()

                inputs, targets = inputs.cuda(), targets.cuda()
                self.optimizer.zero_grad()

                inputs, targets = Variable(inputs), Variable(targets)
                outputs = self.model(inputs)
                outputs = torch.squeeze(outputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.data  # [0]
                total += targets.size(0)

                outputs = outputs.argmax(dim=1)

                correct += outputs.data.eq(targets.data).cpu().sum()

            self.tb.add_scalar("Client:" + str(self.id) + "/Loss", loss, n_round)
            self.tb.add_scalar("Client:" + str(self.id) + "/Correct", correct, n_round)

        print("Finished training for Client: " + str(self.id))
        self.save_local_model(n_round)
        tb.close()

    def save_local_model(self, n_round):
        torch.save(self.model.state_dict(), self.local_model_path
                   + '/local_model_' + str(self.id) + '_round_' + str(n_round) + '.pt')

        torch.save(self.model.state_dict(), self.local_model_path
                   + '/local_model_' + str(self.id) + '.pt')

        print("saved local model of Client: " + str(self.id))
