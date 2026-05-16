"""
main.py — train both models end-to-end and compare results.

Usage:
    python main.py
"""

import os
import torch
from sklearn.metrics import accuracy_score, f1_score

from data import get_loaders
from models import BaselineCNN, ResNetTransfer
from train import train
from evaluate import get_preds, print_metrics, plot_confusion, plot_curves
from gradcam import GradCAM, save_gradcam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {0: "FAKE", 1: "REAL"}


def run_gradcam(model, target_layer, test_loader, name, n=5):
    cam = GradCAM(model, target_layer)
    imgs, labels = next(iter(test_loader))
    for i in range(min(n, len(imgs))):
        x = imgs[i].unsqueeze(0).to(DEVICE)
        pred = model(x).argmax(1).item()
        heatmap = cam(x, class_idx=pred)
        save_gradcam(
            img_tensor=imgs[i],
            heatmap=heatmap,
            true_label=CLASS_NAMES[labels[i].item()],
            pred_label=CLASS_NAMES[pred],
            save_path=f"results/gradcam/{name}_sample_{i}.png",
        )
    print(f"  Grad-CAM saved: results/gradcam/{name}_sample_0..{min(n,len(imgs))-1}.png")


def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    print(f"Device: {DEVICE}\n")

    train_loader, val_loader, test_loader = get_loaders()

    print("Training Baseline CNN")
    baseline = BaselineCNN()
    h_baseline = train(
        baseline, train_loader, val_loader, DEVICE,
        lr=1e-3, epochs=30, patience=5,
        checkpoint="checkpoints/baseline.pt",
    )
    y_true, y_pred = get_preds(baseline, test_loader, DEVICE)
    print_metrics("Baseline CNN", y_true, y_pred)
    plot_curves("baseline_cnn", h_baseline)
    plot_confusion("baseline_cnn", y_true, y_pred)
    baseline_target = list(baseline.features[-1].children())[3]
    run_gradcam(baseline, baseline_target, test_loader, "baseline_cnn")

    print("\nTraining ResNet50 (Transfer Learning)")
    resnet = ResNetTransfer(freeze=True)
    h_resnet = train(
        resnet, train_loader, val_loader, DEVICE,
        lr=3e-4, epochs=30, patience=5,
        checkpoint="checkpoints/resnet50.pt",
        unfreeze_epoch=5,
    )
    y_true_r, y_pred_r = get_preds(resnet, test_loader, DEVICE)
    print_metrics("ResNet50 (Transfer)", y_true_r, y_pred_r)
    plot_curves("resnet50", h_resnet)
    plot_confusion("resnet50", y_true_r, y_pred_r)
    resnet_target = list(resnet.backbone[-4][-1].children())[-3]  # layer2, 4x4 spatial on 32x32 input
    run_gradcam(resnet, resnet_target, test_loader, "resnet50")

    print("\nModel Comparison")
    results = [
        ("Baseline CNN",         y_true,   y_pred),
        ("ResNet50 (Transfer)",  y_true_r, y_pred_r),
    ]
    print(f"  {'Model':<22} {'Accuracy':>10} {'F1-Score':>10}")
    print(f"  {'-'*44}")
    for name, yt, yp in results:
        acc = accuracy_score(yt, yp)
        f1  = f1_score(yt, yp, average="macro")
        print(f"  {name:<22} {acc:>10.4f} {f1:>10.4f}")
    print()


if __name__ == "__main__":
    main()
