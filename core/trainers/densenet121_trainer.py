import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable


class DenseNet121Trainer:
    def __init__(self, model, client_id, ldr, local_model_path):
        '''
        model: model to be trained
        ldr: dataloader
        hp_cfg: configuration of hyperparameters
        '''
        self.id = client_id
        self.model = model
        self.local_model_path = local_model_path

        self.ldr = ldr
        self.batch_size = 8
        self.lr = 0.005
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)

    def train(self):
        train_loss = 0
        total = 0
        correct = 0
        for batch_index, (inputs, targets) in enumerate(self.ldr):

            inputs.cuda()
            targets = torch.FloatTensor(np.array(targets).astype(float)).cuda()

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

        print("Finished training for Client: " + str(self.id))
        self.save_local_model()

    def save_local_model(self):
        torch.save(self.model.state_dict(), self.local_model_path + '/' + str(self.id)
                   + '/local_model_' + str(self.id) + '.pt')

        print("saved local model of Client: " + str(self.id))
