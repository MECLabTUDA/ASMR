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
    keys = ['arch', 'global_model_path', 'data_root', 'agg_method', 'init_model_path']

    if cfg['agg_method'] == 'FedAvgM':
        keys.append('momentum')

    for key in keys:
        server_cfg[key] = cfg[key]

    return server_cfg


def get_experiment_config(cfg):
    experiment_cfg = {}
    keys = ['n_rounds']

    if cfg['agg_method'] == 'FedAvgM':
        keys.append('momentum')

    for key in keys:
        experiment_cfg[key] = cfg[key]

    return experiment_cfg


def get_configs(path):
    cfg = read_config(path)
    cfg['exp_path'] = os.path.join(cfg['root_path'],
                                   f'D_{cfg["data_set"]}_'
                                   f'C_{cfg["n_clients"]}_'
                                   f'E_{cfg["n_local_epochs"]}_'
                                   f'R_{cfg["n_rounds"]}_'
                                   f'Atk_{cfg["fl_attack"]}_'
                                   f'N_{cfg["arch"]}')

    cfg['local_model_root'] = os.path.join(cfg['exp_path'], "clients")
    cfg['global_model_path'] = os.path.join(cfg['exp_path'], "global_model.pt")

    client_cfg = get_client_config(cfg)
    server_cfg = get_server_config(cfg)
    experiment_cfg = get_experiment_config(cfg)

    return client_cfg, server_cfg, experiment_cfg
