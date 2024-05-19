from core.trainers.crc_resnet_trainer import ResnetTrainer
from core.trainers.densenet121_trainer import DenseNet121Trainer
from core.trainers.fcn8_trainer import Fcn8Trainer
from core.trainers.glas_unet_trainer import GlasUnetTrainer
from core.trainers.celeba_resnet101_trainer import Resnet101Trainer

def get_trainer(trainer):
    if trainer == 'densenet121_basic':
        return DenseNet121Trainer
    elif trainer == 'crc_resnet':
        return ResnetTrainer
    elif trainer == 'celeba_resnet':
        return Resnet101Trainer
    pass
