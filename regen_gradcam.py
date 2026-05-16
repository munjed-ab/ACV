"""
regen_gradcam.py — regenerate Grad-CAM overlays from saved checkpoints.

Usage:
    python regen_gradcam.py
"""

import torch
from data import get_loaders
from models import BaselineCNN, ResNetTransfer
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
    print(f"Saved: results/gradcam/{name}_sample_0..{n-1}.png")


_, _, test_loader = get_loaders()

baseline = BaselineCNN()
baseline.load_state_dict(torch.load("checkpoints/baseline_094.pt", map_location=DEVICE, weights_only=True))
baseline.to(DEVICE).eval()
baseline_target = list(baseline.features[-1].children())[3]
run_gradcam(baseline, baseline_target, test_loader, "baseline_cnn")

resnet = ResNetTransfer(freeze=False)
resnet.load_state_dict(torch.load("checkpoints/resnet50_097.pt", map_location=DEVICE, weights_only=True))
resnet.to(DEVICE).eval()
resnet_target = list(resnet.backbone[-4][-1].children())[-3]  # layer2, 4x4 spatial on 32x32 input
run_gradcam(resnet, resnet_target, test_loader, "resnet50")
