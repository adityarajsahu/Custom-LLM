import os
import torch

from src.utils.character_encoder import string_to_integer, integer_to_string, encode, decode
from src.utils.split_dataset import train_val_split, get_batch

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(os.path.join(os.curdir, "data/raw/downloaded_text.txt"), "r", encoding="utf-8") as file:
    text = file.read()

characters = sorted(set(text))
str_to_int_map = string_to_integer(characters)
int_to_str_map = integer_to_string(characters)

data = torch.tensor(encode(text), dtype = torch.long)

train_data, val_data = train_val_split(data, 0.8)
x, y = get_batch(train_data, device)
print("X : ", x)
print("Y : ", y)