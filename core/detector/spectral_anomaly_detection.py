from torch import nn

from utils.data_loaders import get_test_loader
from core.models.get_arch import get_arch


class SpectralVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims):
        super(SpectralVAE, self).__init__()

        #Build encoder
        modules = []
        modules.append(nn.Linear())
        modules.append(nn.Linear())

        self.encoder = nn.Sequential(*modules)

        #Build decoder
        modules = []
        modules.append(nn.Linear())
        modules.append(nn.Linear())

class SpectralDetector:
    def __init__(self):
        pass

    def get_reliable_weights(self):
        pass

    def train_detection_model(self):
        pass

    def get_surrogate_vectors(self, size):
        pass

    def get_reconstruction_error(self):
        pass

    def get_threshold(self):
        pass

    def get_malicious_clients(self):
        pass