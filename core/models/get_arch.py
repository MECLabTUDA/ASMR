import torch
import torchvision.models as torch_classifiers
import segmentation_models_pytorch as smp
from core.models.fcn8 import FCN8s


def get_arch(arch):
    if arch == 'densenet':
        model = torch_classifiers.__dict__['densenet121'](pretrained=False, num_classes=2)
        return model
    elif arch == 'fcn8':
        return FCN8s()
    elif arch == 'unet':
        return smp.Unet(encoder_name='resnet18', decoder_use_batchnorm=True,
                        in_channels=3, classes=1)
    else:
        print('Unknown Model Architecture')
