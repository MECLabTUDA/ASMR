from utils.data_loaders import get_train_loader
from core.models.get_arch import get_arch
from core.trainers.get_trainer import get_trainer


if __name__ == '__main__':
    root_dir = "/local/scratch/camelyon17/camelyon17_v1.0"


    ldr = get_train_loader(root_dir, 32, 2, 1, num_workers=8, pin_memory=True)

    model = get_arch('densenet')

    trainer = get_trainer('densenet121_basic')(model, 1, ldr, '/gris/gris-f/homestud/mikonsta/master-thesis/test/', 5, 1)

    #self, n_round, model, ldr, client_id, fl_attack=None, dp_scale=None):

    trainer.train(1, model, ldr, 1, 'sfa')

