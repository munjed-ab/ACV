import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report,
)


@torch.no_grad()
def get_preds(model, loader, device):
    model.eval()
    preds, labels = [], []
    for imgs, lbl in loader:
        preds.extend(model(imgs.to(device)).argmax(1).cpu().tolist())
        labels.extend(lbl.tolist())
    return labels, preds


def print_metrics(name, y_true, y_pred):
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  F1-Score : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["FAKE", "REAL"]))


def plot_confusion(name, y_true, y_pred, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
    plt.title(f"{name} — Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}_confusion.png", dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/{name}_confusion.png")


def plot_curves(name, history, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(ep, history["train_loss"], label="Train")
    ax1.plot(ep, history["val_loss"],   label="Val")
    ax1.set_title(f"{name} — Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(ep, history["train_acc"], label="Train")
    ax2.plot(ep, history["val_acc"],   label="Val")
    ax2.set_title(f"{name} — Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}_curves.png", dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/{name}_curves.png")
