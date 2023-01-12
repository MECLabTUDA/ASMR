import torch.nn.functional as F

def get_update_cosine_sim(global_model, local_models):
    cos_sim = {}
    g_model = prepare_weights(global_model)
    for lmodel in local_models.keys():
             cos_sim[lmodel] = F.cosine_similarity(g_model, prepare_weights(local_models[lmodel]))
    return cos_sim


def prepare_weights(weight):
    pass

def detect_mal_clients(cos_sims, threshold):
    pass
