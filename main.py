import os
import torch

from src.utils.character_encoder import string_to_integer, integer_to_string, encode, decode

with open(os.path.join(os.curdir, "data/raw/downloaded_text.txt"), "r", encoding="utf-8") as file:
    text = file.read()

characters = sorted(set(text))
str_to_int_map = string_to_integer(characters)
int_to_str_map = integer_to_string(characters)

data = torch.tensor(encode(text), dtype = torch.long)
print(data[:100])