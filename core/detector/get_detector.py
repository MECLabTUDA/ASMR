from core.detector.clustering import ClusteringDetector
from core.detector.dnc import Dnc
from core.detector.krum import Krum
from core.detector.asmr import ASMR
from core.detector.opt_detector import Optimal_detector


def get_detector(detector):
    if detector == 'krum':
        return Krum
    elif detector == 'clustering':
        return ClusteringDetector
    elif detector == 'dnc':
        return Dnc
    elif detector == 'asmr':
        return ASMR
    elif detector == 'optimal':
        return Optimal_detector
    else:
        return None

