from core.detector.krum import krum_detection


def get_detector(detector):
    match detector:
        case 'krum':
            return krum_detection
