import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
block_size = 256
max_iter = 3000
learning_rate = 3e-4
eval_iters = 250
n_embed = 512
n_head = 8
n_layer = 6
dropout = 0.2

data_path = "data/downloaded_text.txt"