from core.detector.hog import MudHog
from core.detector.krum import Krum
from core.detector.spectral_anomaly_detection import SpectralAnomaly


def get_detector(detector):
    if detector == 'krum':
        return Krum
    elif detector == 'mud-hog':
        return MudHog
    elif detector == 'spectral':
        return SpectralAnomaly
    else:
        return None

