from core.trainers.densenet121_trainer import DenseNet121Trainer


def get_trainer(trainer):
    if trainer == 'densenet121_basic':
        return DenseNet121Trainer
    pass
