"""Tüm deneyleri çalıştırır, sonuçları ve grafikleri results/ klasörüne kaydeder.

Çalıştırma: python src/experiments.py

Kural: Model seçimi yalnızca validation doğruluğuna göre yapılır. Test seti eğitim ve
seçim bittikten sonra, her model için yalnızca bir kez değerlendirilir.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from data import get_loaders
from train import evaluate, run

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODELS = RES / "models"

BASE = {"hidden": [256, 128], "dropout": 0.0, "optimizer": "adam", "lr": 1e-3,
        "batch_size": 64, "epochs": 15}


def cfg(name, group, **changes):
    return {**BASE, **changes, "name": name, "group": group}


EXPERIMENTS = [
    # Deney 1 - Temel MLP: 2 gizli katman, ReLU, CrossEntropy, Adam
    cfg("baseline", "Deney 1 - Temel model"),
    # Deney 2 - Mimari: derinlik ve genişlik
    cfg("small_128-64", "Deney 2 - Mimari", hidden=[128, 64]),
    cfg("wide_512-256", "Deney 2 - Mimari", hidden=[512, 256]),
    cfg("deep_512-256-128", "Deney 2 - Mimari", hidden=[512, 256, 128]),
    # Deney 3 - Optimizer ve öğrenme oranı
    cfg("sgd_lr0.01", "Deney 3 - Optimizer / LR", optimizer="sgd", lr=0.01),
    cfg("sgd_momentum_lr0.01", "Deney 3 - Optimizer / LR", optimizer="sgd_momentum", lr=0.01),
    cfg("adam_lr0.01", "Deney 3 - Optimizer / LR", lr=1e-2),
    cfg("adam_lr0.0001", "Deney 3 - Optimizer / LR", lr=1e-4),
    # Deney 4 - Düzenlileştirme ve batch size
    cfg("dropout0.2", "Deney 4 - Dropout / Batch", dropout=0.2),
    cfg("dropout0.5", "Deney 4 - Dropout / Batch", dropout=0.5),
    cfg("batch256", "Deney 4 - Dropout / Batch", batch_size=256),
]


def plot_curves(results, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for r in results:
        h, name = r["history"], r["config"]["name"]
        ep = range(1, len(h["train_loss"]) + 1)
        (line,) = axes[0].plot(ep, h["train_loss"], "--", alpha=0.7)
        axes[0].plot(ep, h["val_loss"], "-", color=line.get_color(), label=name)
        axes[1].plot(ep, h["train_acc"], "--", color=line.get_color(), alpha=0.7)
        axes[1].plot(ep, h["val_acc"], "-", color=line.get_color(), label=name)
    axes[0].set(title="Loss (kesikli: train, düz: validation)", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Accuracy (kesikli: train, düz: validation)", xlabel="Epoch", ylabel="Accuracy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_summary(rows, path):
    names = [r["name"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = range(len(rows))
    ax.bar([i - 0.2 for i in x], [r["best_val_acc"] * 100 for r in rows], 0.4, label="Validation")
    ax.bar([i + 0.2 for i in x], [r["test_acc"] * 100 for r in rows], 0.4, label="Test")
    ax.set_xticks(list(x), names, rotation=30, ha="right")
    ax.set_ylim(90, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Yapılandırmalara göre doğruluk")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    RES.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    all_results = []
    for c in EXPERIMENTS:
        print(f"\n=== {c['group']}: {c['name']} ===")
        model, result = run(c)
        torch.save(model.state_dict(), MODELS / f"{c['name']}.pt")
        all_results.append((model, result))

    # Model seçimi: SADECE validation doğruluğu
    best_model, best = max(all_results, key=lambda mr: mr[1]["best_val_acc"])
    print(f"\nValidation'a göre en iyi model: {best['config']['name']}")

    # Eğitim ve seçim bitti -> test seti her model için bir kez değerlendirilir
    _, _, test_loader = get_loaders()
    loss_fn = nn.CrossEntropyLoss()
    rows = []
    for model, r in all_results:
        test_loss, test_acc = evaluate(model, test_loader, loss_fn)
        h, b = r["history"], r["best_epoch"] - 1
        rows.append({
            "name": r["config"]["name"], "group": r["config"]["group"],
            "hidden": r["config"]["hidden"], "dropout": r["config"]["dropout"],
            "optimizer": r["config"]["optimizer"], "lr": r["config"]["lr"],
            "batch_size": r["config"]["batch_size"], "epochs": r["config"]["epochs"],
            "params": r["params"], "best_epoch": r["best_epoch"],
            "train_loss": h["train_loss"][b], "train_acc": h["train_acc"][b],
            "val_loss": h["val_loss"][b], "best_val_acc": r["best_val_acc"],
            "test_loss": test_loss, "test_acc": test_acc,
            "train_time_sec": r["train_time_sec"],
        })

    (RES / "results.json").write_text(json.dumps(
        {"best_by_validation": best["config"]["name"], "rows": rows,
         "histories": {r["config"]["name"]: r["history"] for _, r in all_results}},
        indent=2, ensure_ascii=False))
    torch.save(best_model.state_dict(), MODELS / "best.pt")
    (MODELS / "best.json").write_text(json.dumps(best["config"], indent=2))

    groups = {}
    for _, r in all_results:
        groups.setdefault(r["config"]["group"], []).append(r)
    base = all_results[0][1]
    for i, (g, rs) in enumerate(groups.items(), 1):
        if base not in rs:
            rs = [base] + rs  # karşılaştırma için temel modeli de çiz
        plot_curves(rs, g, RES / f"deney{i}_egriler.png")
    plot_summary(rows, RES / "ozet_dogruluk.png")

    print("\n{:<22} {:>9} {:>9} {:>9} {:>8}".format("model", "val_acc", "test_acc", "params", "süre(s)"))
    for r in rows:
        print("{:<22} {:>9.4f} {:>9.4f} {:>9} {:>8.1f}".format(
            r["name"], r["best_val_acc"], r["test_acc"], r["params"], r["train_time_sec"]))


if __name__ == "__main__":
    main()
