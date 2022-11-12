from core.aggregation.FedAvg import FedAvg


def get_aggregation(method):
    """
    returns aggregation method
    """
    if method == 'FedAvg':
        return FedAvg
    pass
