from copy import deepcopy

import torch


def net2vec(net) -> (torch.Tensor):
    param_float = getFloatSubModules(net)

    components = []
    for param in param_float:
        components.append(net[param])
    vec = torch.cat([component.flatten() for component in components])
    return vec


def vec2net(vec: torch.Tensor, net) -> None:
    '''
    convert a 1 dimension Tensor to state dict

    vec : torch vector with shape([d]), d is the number of elements \
            in all module components specified in `param_name`
    net : the state dict to hold the value

    return
    None
    '''
    param_float = getFloatSubModules(net)
    shapes, sizes = getNetMeta(net)
    partition = list(sizes[param] for param in param_float)
    flattenComponents = dict(zip(param_float, torch.split(vec, partition)))
    components = dict(((k, v.reshape(shapes[k])) for (k, v) in flattenComponents.items()))
    net.update(components)
    return net


def stackStateDicts(deltas):
    '''
    stacking a list of state_dicts to a state_dict of stacked states, ignoring non float values

    deltas: [dict, dict, dict, ...]
        for all dicts, they have the same keys and different values in the form of torch.Tensor with shape s, e.g. s=torch.shape(10,10)

    return
        stacked: dict
            it has the same keys as the dict in deltas, the value is a stacked flattened tensor from the corresponding tenors in deltas.
            e.g. deltas[i]["conv.weight"] has a shape torch.shape(10,10),
                then stacked["conv.weight"]] has shape torch.shape(10*10,n), and
                stacked["conv.weight"]][:,i] is equal to deltas[i]["conv.weight"].flatten()
    '''
    stacked = deepcopy(deltas[0])
    for param in stacked:
        stacked[param] = None
    for param in stacked:
        param_stack = torch.stack([delta[param] for delta in deltas], -1)
        shaped = param_stack.view(-1, len(deltas))
        stacked[param] = shaped
    return stacked


def getFloatSubModules(Delta) -> list:
    param_float = []
    for param in Delta:
        #if not "FloatTensor" in Delta[param].type():
        #    continue
        param_float.append(param)
    return param_float


def getNetMeta(Delta) -> (dict, dict):
    '''
    get the shape and number of elements in each modules of Delta
    get the module components of type float and otherwise
    '''
    shapes = dict(((k, v.shape) for (k, v) in Delta.items()))
    sizes = dict(((k, v.numel()) for (k, v) in Delta.items()))
    return shapes, sizes
