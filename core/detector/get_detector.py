from core.detector.hog import MudHog
from core.detector.krum import Krum
from core.detector.spectral_anomaly_detection import SpectralAnomaly


def get_detector(detector):
    match detector:
        case 'krum':
            return Krum
        case 'mud-hog':
            return MudHog
        case 'spectral':
            return SpectralAnomaly
