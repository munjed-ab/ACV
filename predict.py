"""
predict.py — run inference on any image using trained checkpoints.

Tiles the image into 32x32 patches, predicts each, aggregates results.
Visualization saved to results/predictions/<name>_<model>.png

Usage:
    python predict.py image.jpg
    python predict.py image.jpg --model baseline
    python predict.py image.jpg --model resnet --patch-size 64
"""

import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms

from models import BaselineCNN, ResNetTransfer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {0: "FAKE", 1: "REAL"}
CHECKPOINTS = {
    "baseline": "checkpoints/baseline_094.pt",
    "resnet":   "checkpoints/resnet50_097.pt",
}

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


def load_model(name):
    if name == "baseline":
        model = BaselineCNN()
    else:
        model = ResNetTransfer(freeze=False)
    model.load_state_dict(
        torch.load(CHECKPOINTS[name], map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE).eval()
    return model


def tile_image(img: Image.Image, patch_size=32):
    """Split PIL image into (patch_size x patch_size) patches with their grid positions."""
    img = img.convert("RGB")
    w, h = img.size
    patches, positions = [], []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append(TRANSFORM(patch))
            positions.append((x, y))
    return patches, positions, w, h


@torch.no_grad()
def predict_patches(model, patches, batch_size=256):
    probs_all = []
    for i in range(0, len(patches), batch_size):
        batch = torch.stack(patches[i:i+batch_size]).to(DEVICE)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_all.append(probs)
    return np.concatenate(probs_all, axis=0)  # (N, 2)


def visualize(img, positions, probs, w, h, patch_size, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # left: original
    axes[0].imshow(img.convert("RGB"))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # right: patch heatmap overlay
    axes[1].imshow(img.convert("RGB"))
    axes[1].set_title("Patch Predictions  (green=REAL, red=FAKE)")
    axes[1].axis("off")

    for (x, y), prob in zip(positions, probs):
        fake_prob = prob[0]
        color = "red" if fake_prob > 0.5 else "green"
        alpha = 0.3 + 0.4 * abs(fake_prob - 0.5) * 2  # stronger color = more confident
        rect = mpatches.Rectangle(
            (x, y), patch_size, patch_size,
            linewidth=0.5, edgecolor=color, facecolor=color, alpha=alpha,
        )
        axes[1].add_patch(rect)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", choices=["baseline", "resnet"], default="resnet")
    parser.add_argument("--patch-size", type=int, default=32)
    args = parser.parse_args()


    import os
    img = Image.open(args.image)
    w, h = img.size
    print(f"Image: {args.image}  ({w}x{h})")
    print(f"Model: {args.model}")

    patches, positions, w, h = tile_image(img, patch_size=args.patch_size)

    if len(patches) == 0:
        print(f"Image too small — needs at least {args.patch_size}x{args.patch_size} pixels.")
        sys.exit(1)

    print(f"Patches: {len(patches)}  ({w // args.patch_size} cols x {h // args.patch_size} rows)")

    model = load_model(args.model)
    probs = predict_patches(model, patches)

    fake_probs = probs[:, 0]
    mean_fake  = fake_probs.mean()
    mean_real  = 1 - mean_fake
    verdict    = "FAKE" if mean_fake > 0.5 else "REAL"
    confidence = max(mean_fake, mean_real) * 100
    fake_patch_pct = (fake_probs > 0.5).mean() * 100

    print()
    print(f"  Verdict    : {verdict}")
    print(f"  Confidence : {confidence:.1f}%")
    print(f"  FAKE score : {mean_fake:.4f}  |  REAL score : {mean_real:.4f}")
    print(f"  FAKE patches: {fake_patch_pct:.1f}%  of {len(patches)} total")

    os.makedirs("results/predictions", exist_ok=True)
    name = args.image.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    save_path = f"results/predictions/{name}_{args.model}.png"

    visualize(img, positions, probs, w, h, args.patch_size, save_path)


if __name__ == "__main__":
    main()
