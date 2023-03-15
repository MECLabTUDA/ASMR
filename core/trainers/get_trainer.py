from core.trainers.densenet121_trainer import DenseNet121Trainer
from core.trainers.fcn8_trainer import Fcn8Trainer


def get_trainer(trainer):
    if trainer == 'densenet121_basic':
        return DenseNet121Trainer
    elif trainer == 'fcn8':
        return Fcn8Trainer
    pass
