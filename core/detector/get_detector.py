from core.detector.clustering import ClusteringDetector
from core.detector.dnc import Dnc
from core.detector.hog import MudHog
from core.detector.krum import Krum
from core.detector.mirko_detector import MirkoDetector
from core.detector.spectral_anomaly_detection import SpectralAnomaly


def get_detector(detector):
    if detector == 'krum':
        return Krum
    elif detector == 'mud-hog':
        return MudHog
    elif detector == 'spectral':
        return SpectralAnomaly
    elif detector == 'mirko':
        return MirkoDetector
    elif detector == 'clustering':
        return ClusteringDetector
    elif detector == 'dnc':
        return Dnc
    else:
        return None

