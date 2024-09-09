import torch 
from config import device, block_size, data_path

def map_character_to_integer(characters):
    str_to_int = {}
    for i, ch in enumerate(characters):
        str_to_int[ch] = i
    return str_to_int

def map_integer_to_character(characters):
    int_to_str = {}
    for i, ch in enumerate(characters):
        int_to_str[i] = ch
    return int_to_str

def encode(s, str_to_int):
    encoded_list = []
    for ch in s:
        encoded_list.append(str_to_int[ch])
    return encoded_list

def decode(encoded_list, int_to_str):
    s = ""
    for i in encoded_list:
        s += int_to_str[i]
    return s

def load_data():
    with open(data_path, "r", encoding = "utf-8") as file:
        text = file.read()
    characters = sorted(set(text))
    vocab_size = len(characters)

    str_to_int = map_character_to_integer(characters)
    int_to_str = map_integer_to_character(characters)

    data = torch.tensor(encode(text, str_to_int), dtype = torch.long)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    return train_data, val_data, vocab_size

def get_batch(data, batch_size):
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in indices]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in indices]).to(device)
    return x, y