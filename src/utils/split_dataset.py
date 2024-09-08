import torch
from src.utils.config_loader import load_config

def train_val_split(data, train_size = 0.8):
    train_len = int(train_size * len(data))
    train_data = data[:train_len]
    val_data = data[train_len:]

    return train_data, val_data

def get_batch(data, device = "cpu"):
    config = load_config()
    indices = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    x = torch.stack([data[i : i + config["block_size"]] for i in indices]).to(device)
    y = torch.stack([data[i + 1 : i + config["block_size"] + 1] for i in indices]).to(device)
    return x, y