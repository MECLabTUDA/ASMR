import yaml

def read_config(path):
    return yaml.load(open(path), Loader=yaml.Loader)['experiment']