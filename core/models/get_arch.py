import torch
import torchvision.models as torch_classifiers
from torch import nn

from torchvision.models import resnet50, resnet101



def get_arch(arch):
    if arch == 'densenet':
        model = torch_classifiers.densenet121(pretrained=False)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 2)
        return model
    elif arch == 'resnet50':
        model = resnet50()
        model.fc = nn.Linear(2048, 9)
        return model
    elif arch == 'resnet101':
        model = resnet101(2)
        return model
        print('Unknown Model Architecture')
