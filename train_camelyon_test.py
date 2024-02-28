#!/usr/bin/python3

import torch
from torch import optim, nn
from torch.autograd import Variable
from tqdm import tqdm

from core.models.get_arch import get_arch
from utils.data_loaders import get_test_loader


if __name__ == '__main__':

    ldr = get_test_loader("/local/scratch/camelyon17/camelyon17_v1.0", 64, 'camelyon17')

    model = get_arch('densenet')

    model.load_state_dict(torch.load('/gris/gris-f/homestud/mikonsta/master-thesis/FedPath/store/init_models/densenet121.pt'))

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    device = 'cuda:0'
    criterion = nn.CrossEntropyLoss()
    model.train()

    train_loss = 0
    total = 0
    correct = 0
    model.to(device)
    epoch_loss = []

    for epoch in range(50):
        batch_loss = []

        for batch_index, (inputs, targets) in enumerate(tqdm(ldr), 0):
            # inputs.cuda()

            # targets = torch.FloatTensor(np.array(targets).astype(float)).cuda()

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs, targets)

            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=5.0)

            optimizer.step()

            train_loss += loss.data  # [0]
            total += targets.size(0)

            outputs = outputs.argmax(dim=1)

            correct += outputs.data.eq(targets.data).cpu().sum()
            batch_loss.append(loss.item())

        if len(batch_loss) > 0:
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print(
                'Local Training Epoch: {} \t Loss: {:.6f} \t correct: {}, train_acc: {:.3f}'.format(
                    epoch,
                    sum(
                        epoch_loss) / len(
                        epoch_loss), correct, 100. * correct / total))

    # print(f"Finished training for Client:{self.id}, loss:{loss}, " + str(self.id))
        torch.save(model.state_dict(), f'/gris/gris-f/homestud/mikonsta/master-thesis/spectral_weights/w_{epoch}.pt')
    # self.tb.close()






