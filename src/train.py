"""Eğitim ve değerlendirme döngüsü."""
import time

import torch
from torch import nn

from data import get_loaders
from model import MLP, count_params


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            total_loss += loss_fn(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def make_optimizer(name, params, lr):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Bilinmeyen optimizer: {name}")


def run(config, seed=42, verbose=True):
    """Bir yapılandırmayı eğitir. Test seti burada KULLANILMAZ; sadece train ve validation.

    config: {"name", "hidden", "dropout", "optimizer", "lr", "batch_size", "epochs"}
    """
    torch.manual_seed(seed)
    train_loader, val_loader, _ = get_loaders(config["batch_size"], seed=seed)
    model = MLP(config["hidden"], config.get("dropout", 0.0))
    loss_fn = nn.CrossEntropyLoss()
    opt = make_optimizer(config["optimizer"], model.parameters(), config["lr"])

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val, best_state = -1.0, None
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        history["train_loss"].append(total_loss / n)
        history["train_acc"].append(correct / n)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        # en iyi validation doğruluğuna sahip ağırlıkları sakla
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  [{config['name']}] epoch {epoch:2d}/{config['epochs']}  "
                  f"train_loss={total_loss / n:.4f} train_acc={correct / n:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    train_time = time.perf_counter() - start

    model.load_state_dict(best_state)
    return model, {
        "config": config,
        "params": count_params(model),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
        "train_time_sec": train_time,
    }
