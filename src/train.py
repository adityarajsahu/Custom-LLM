import torch
from model import GPTLanguageModel
from dataset import load_data, get_batch
from utils import estimate_loss
from config import device, max_iter, eval_iters, learning_rate, batch_size

if __name__ == "__main__":
    train_data, val_data, vocab_size = load_data()

    model = GPTLanguageModel(vocab_size)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    for iter in range(max_iter):
        if iter % eval_iters == 0:
            losses = estimate_loss(model, train_data, val_data)
            print("step : {}, train loss : {:.3f}, val loss : {:.3f}".format(iter, losses["train"], losses["val"]))
        
        x, y = get_batch(train_data, batch_size)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

    print(loss.item())