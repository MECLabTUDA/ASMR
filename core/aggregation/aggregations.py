import logging

from core.aggregation.fedAvg import FedAvg
from core.aggregation.fedAvgM import FedAvgM

def get_aggregation(method):
    """
    returns aggregation method
    """
    if method == 'FedAvg':
        return FedAvg
    if method == 'FedAvgM':
        return FedAvgM
    else:
        logging.error("Unknown aggregation method")
