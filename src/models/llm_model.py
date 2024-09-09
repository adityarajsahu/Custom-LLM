import torch
import torch.nn as nn
from torch.nn import functional as F

from src.utils.config_loader import load_config

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.config = load_config()
        self.key = nn.Linear(self.config["n_embed"], head_size, bias = False)
        self.query = nn.Linear(self.config["n_embed"], head_size, bias = False)
        self.value = nn.Linear(self.config["n_embed"], head_size, bias = False)
        self.register_buffer("tril", torch.tril(torch.ones(self.config["block_size"], self.config["block_size"])))
        self.dropout = nn.Dropout(self.config["dropout"])
    
    def forward(self, x):
        batch, no_of_embeddings, embed_size = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        w = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        w = w.masked_fill(self.tril[:no_of_embeddings, :no_of_embeddings] == 0, float("-inf"))
        w = F.softmax(w, dim = -1)
        w = self.dropout(w)
        out = w @ v
        return out