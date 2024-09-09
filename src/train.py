import torch
from tqdm import tqdm
from model import GPTLanguageModel
from dataset import load_data, get_batch
from utils import estimate_loss
from config import device, max_iter, learning_rate, batch_size, best_model_path, last_model_path

if __name__ == "__main__":
    train_data, val_data, vocab_size = load_data()

    model = GPTLanguageModel(vocab_size)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: {:.2f}M".format(total_params / 1e6))

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    best_val_loss = float("inf")

    progress_bar = tqdm(range(max_iter), desc = "Training", unit = "step", ncols = 100)
    
    for iter in progress_bar:
        losses = estimate_loss(model, train_data, val_data)

        progress_bar.set_postfix({
            "train loss": "{:.3f}".format(losses["train"]),
            "val loss": "{:.3f}".format(losses["val"])
        })

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(model.state_dict(), best_model_path)
            progress_bar.write("Best model saved with validation loss: {:.3f}".format(best_val_loss))
        
        x, y = get_batch(train_data, batch_size)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), last_model_path)
    print(loss.item())