import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._act = self._grad = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_act", o.detach())
        )
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grad", go[0].detach())
        )

    def __call__(self, x, class_idx=None):
        self.model.eval()
        out = self.model(x)
        idx = class_idx if class_idx is not None else out.argmax(1).item()
        self.model.zero_grad()
        out[0, idx].backward()

        weights = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._act).sum(dim=1, keepdim=True)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def save_gradcam(img_tensor, heatmap, true_label, pred_label, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = (img_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(img);                                           axes[0].set_title("Original")
    axes[1].imshow(heatmap, cmap="jet");                           axes[1].set_title("Grad-CAM")
    axes[2].imshow(img); axes[2].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[2].set_title(f"True: {true_label}  |  Pred: {pred_label}")
    for ax in axes: ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
