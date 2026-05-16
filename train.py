import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def train(model, train_loader, val_loader, device,
          lr=1e-3, epochs=30, patience=5, checkpoint="best.pt",
          unfreeze_epoch=None):

    model.to(device)
    opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, no_improve = float("inf"), 0

    for epoch in range(1, epochs + 1):
        if unfreeze_epoch and epoch == unfreeze_epoch:
            model.unfreeze()
            # rebuild optimizer to pick up newly unfrozen params at lower LR
            opt = Adam(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)

        for phase, loader, is_train in [("train", train_loader, True), ("val", val_loader, False)]:
            model.train(is_train)
            total_loss, correct, n = 0.0, 0, 0
            with torch.set_grad_enabled(is_train):
                for imgs, labels in tqdm(loader, desc=f"Epoch {epoch:03d} [{phase}]", leave=False):
                    imgs, labels = imgs.to(device), labels.to(device)
                    out = model(imgs)
                    loss = criterion(out, labels)
                    if is_train:
                        opt.zero_grad(); loss.backward(); opt.step()
                    total_loss += loss.item() * len(labels)
                    correct += (out.argmax(1) == labels).sum().item()
                    n += len(labels)
            history[f"{phase}_loss"].append(total_loss / n)
            history[f"{phase}_acc"].append(correct / n)

        sched.step()
        print(f"Epoch {epoch:03d} | "
              f"train_loss={history['train_loss'][-1]:.4f}  train_acc={history['train_acc'][-1]:.4f} | "
              f"val_loss={history['val_loss'][-1]:.4f}  val_acc={history['val_acc'][-1]:.4f}")

        val_loss = history["val_loss"][-1]
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return history
