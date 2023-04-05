import copy


def flip_signs(w, scale=-1):
    w_attacked = copy.deepcopy(w)
    for k in w_attacked.keys():
        w_attacked[k] = scale * w_attacked[k].float()
    return w_attacked
