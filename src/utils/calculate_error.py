import torch
from src.utils.config_loader import load_config
from src.utils.split_dataset import get_batch

@torch.no_grad()
def estimate_loss():
    config = load_config()

    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out