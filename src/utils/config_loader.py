import yaml

def load_config(key = "params", config_path = "./config/train_config.yaml"):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config.get(key, {})