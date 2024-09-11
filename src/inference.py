import torch
from dataset import encode, decode
from model import GPTLanguageModel
import config

model = GPTLanguageModel(67)
model.load_state_dict(torch.load(config.best_model_path, weights_only = True))
model.eval()
model.to(config.device)

prompt = "Hello!, How are you?"
context = torch.tensor(encode(prompt, config.str_to_int), dtype = torch.long, device = config.device)

with torch.no_grad():
    generate_indices = model.generate(context.unsqueeze(0), max_new_tokens = 100)[0].tolist()
generated_text = decode(generate_indices, config.int_to_str)
print(generated_text)