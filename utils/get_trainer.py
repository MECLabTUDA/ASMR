from core.trainers.densenet121_trainer import DenseNet121Trainer


def get_trainer(trainer, model, ldr, hp_cfg, local_model_path):
    if trainer == 'densenet121_basic':
        return DenseNet121Trainer(model, ldr, hp_cfg, local_model_path)
    pass
