import torch
import torch.nn as nn
import torch.nn.functional as F
from config import block_size, n_embed, n_head, n_layer, dropout, device

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, num_embeddings, embed_size = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        w = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        w = w.masked_fill(self.tril[:num_embeddings, :num_embeddings] == 0, float("-inf"))
        w = F.softmax(w, dim = -1)
        w = self.dropout(w)
        out = w @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embed // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward_network = FeedForward()
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        y = self.self_attention(x)
        x = self.layer_norm1(x + y)
        y = self.feed_forward_network(x)
        x = self.layer_norm2(x + y)
        return x
    
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.linear = nn.Linear(n_embed, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, index, targets = None):
        batch, num_embeddings = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(num_embeddings, device = device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.linear(x)

        if targets is None:
            loss = None
        else:
            batch, num_embeddings, embed_size = logits.shape
            logits = logits.view(batch * num_embeddings, embed_size)
            targets = targets.view(batch * num_embeddings)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            index_next = torch.multinomial(probs, num_samples = 1)
            index = torch.cat((index, index_next), dim = 1)
        return index