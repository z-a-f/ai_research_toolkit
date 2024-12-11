import yaml

DEFAULT_CONFIG = {
    'training': {
        'accelerator': 'cpu',
        'devices': 'auto',
    }
}

def with_default(default_config):
    # default_config must be a dictionary
    # func must return a dictionary
    def decorator(func):
        def wrapper(*args, **kwargs):
            loaded_cfg = func(*args, **kwargs)
            nodes = [(loaded_cfg, DEFAULT_CONFIG)]
            while nodes:
                loaded, default = nodes.pop()
                for k, v in default.items():
                    if k not in loaded:
                        loaded[k] = v
                    elif isinstance(v, dict):
                        nodes.append((loaded[k], v))
            return loaded_cfg
        return wrapper
    return decorator

@with_default(DEFAULT_CONFIG)
def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg