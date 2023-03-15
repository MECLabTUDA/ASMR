import torchvision.models as torch_classifiers

from core.models.fcn8 import FCN8s


def get_arch(arch):
    if arch == 'densenet':
        model = torch_classifiers.__dict__['densenet121'](pretrained=False, num_classes=2)
        return model
    elif arch == 'fcn8':
        return FCN8s()
    else:
        print('Unknown Model Architecture')
