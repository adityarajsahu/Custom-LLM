import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
block_size = 128
max_iter = 3000
learning_rate = 3e-4
eval_iters = 250
n_embed = 512
n_head = 8
n_layer = 6
dropout = 0.2

data_path = "data/downloaded_text.txt"
best_model_path = "checkpoints/best_model.pt"
last_model_path = "checkpoints/last_model.pt"