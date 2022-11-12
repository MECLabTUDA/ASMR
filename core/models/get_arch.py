import torchvision.models as torch_classifiers


def get_arch(arch):
    if arch == 'densenet':
        model = torch_classifiers.__dict__['densenet121'](pretrained=False, num_classes=2)
        return model
    else:
        print('Unknown Model Architecture')
