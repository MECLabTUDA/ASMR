import copy


def flig_signs(w, scale):
    w_attacked = copy.deepcopy(w)
    for k in w_attacked.keys():
        w_attacked[k] = scale * w_attacked[k].float()
    return w_attacked
