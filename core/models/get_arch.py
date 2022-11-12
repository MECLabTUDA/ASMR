import torchvision.models as torch_classifiers


def get_arch(arch, pretrained):
    if arch == 'densenet':
        model = torch_classifiers.__dict__['densenet121'](pretrained=pretrained, num_classes=2)
        return model
    pass
