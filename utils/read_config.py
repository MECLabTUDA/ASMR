import yaml

def read_config(path):
    return yaml.load(open(path), Loader=yaml.Loader)


def get_client_config(cfg):
    client_cfg = {}
    keys = ['arch', 'local_model_root', 'global_model_path', 'trainer', 'n_clients',
            'data_root', 'batch_size', 'n_local_epochs']
    for key in keys:
        client_cfg[key] = cfg[key]

    return client_cfg


def get_server_config(cfg):
    server_cfg = {}
    keys = ['arch', 'global_model_path', 'data_root', 'agg_method']

    if cfg['agg_method'] == 'FedAvgM':
        keys.append('momentum')

    for key in keys:
        server_cfg[key] = cfg[key]

    return server_cfg


def get_configs(path):
    cfg = read_config(path)
    client_cfg = get_client_config(cfg)
    server_cfg = get_server_config(cfg)

    return client_cfg, server_cfg

