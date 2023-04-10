import torch
import torchvision.models as torch_classifiers
import segmentation_models_pytorch as smp
from torch import nn

from core.models.fcn8 import FCN8s
from torchvision.models import resnet50

def get_arch(arch):
    if arch == 'densenet':
        model = torch_classifiers.__dict__['densenet121'](pretrained=False, num_classes=2)
        return model
    elif arch == 'fcn8':
        return FCN8s()
    elif arch == 'unet':
        return smp.Unet(encoder_name='resnet18', decoder_use_batchnorm=True,
                        in_channels=3, classes=1)
    elif arch == 'resnet50':
        model = resnet50()
        model.fc = nn.Linear(2048, 9)
        return model
    else:
        print('Unknown Model Architecture')
