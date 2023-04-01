# code from: https://github.com/cpwan/Attack-Adaptive-Aggregation-in-Federated-Learning/blob/runx/rules/multiKrum.py

import torch
import torch.nn as nn

from core.detector.helpers import net2vec

'''
Krum aggregation
- find the point closest to its neignborhood
Reference:
Blanchard, Peva, Rachid Guerraoui, and Julien Stainer. "Machine learning with adversaries: Byzantine tolerant gradient descent." Advances in Neural Information Processing Systems. 2017.
`https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf`
'''


def krum_detection(clients_info, k):
    clients = [{'id': client['id'], 'weights': net2vec(client['weights'].state_dict())} for client in clients_info]

    clients = sorted(clients, key=lambda client: client['id'])

    vecs = [c['weights'] for c in clients]

    stackedVecs = torch.stack(vecs, 1).unsqueeze(0)

    x = stackedVecs.permute(0, 2, 1)
    cdist = torch.cdist(x, x, p=2)

    nbhDist, nbh = torch.topk(cdist, k, largest=False)
    # Closest Vector to all others
    i_star = torch.argmin(nbhDist.sum(2))

    valid_clients = nbh[:, i_star, :]

    print(valid_clients)

    malicious_clients = []
    benign_clients = []
    for client in clients_info:

        if client['id'] in valid_clients:
            benign_clients.append(client)
        else:
            malicious_clients.append(client['id'])

    return benign_clients, malicious_clients
