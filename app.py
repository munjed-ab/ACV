"""
app.py Gradio Space for SynthSentinel AI image detector.

Accepts full-resolution images, tiles into 32×32 patches (matching training
distribution), classifies each patch, and aggregates into a final verdict.

Domain shift warning: models were trained on CIFAKE (32×32 CIFAR-10 real +
Stable Diffusion v1.4 fake). Performance on other generators or high-res
natural photos will differ from the reported 94–97% test accuracy.
"""

import io
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr
from PIL import Image
from torchvision import transforms

from models import BaselineCNN, ResNetTransfer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "ResNet50 (97.8% acc)": "checkpoints/resnet50_097.pt",
    "Baseline CNN (94.5% acc)": "checkpoints/baseline_094.pt",
}
MODEL_CLASSES = {
    "ResNet50 (97.8% acc)": lambda: ResNetTransfer(freeze=False),
    "Baseline CNN (94.5% acc)": BaselineCNN,
}

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

_cache: dict = {}


def load_model(name: str):
    if name not in _cache:
        model = MODEL_CLASSES[name]()
        state = torch.load(CHECKPOINTS[name], map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        _cache[name] = model
    return _cache[name]


def tile_image(img: Image.Image, patch_size: int = 32):
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
def predict_patches(model, patches, batch_size: int = 512):
    results = []
    for i in range(0, len(patches), batch_size):
        batch = torch.stack(patches[i : i + batch_size]).to(DEVICE)
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        results.append(probs)
    return np.concatenate(results, axis=0)  # (N, 2): col0=FAKE, col1=REAL


def render_heatmap(img: Image.Image, positions, probs, patch_size: int) -> Image.Image:
    w, h = img.size
    dpi = 100
    fig_w = max(12, w / dpi * 2 + 2)
    fig_h = max(5, h / dpi + 1)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    axes[0].imshow(img.convert("RGB"))
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(img.convert("RGB"))
    axes[1].set_title("Patch map  (green = REAL · red = FAKE)", fontsize=13)
    axes[1].axis("off")

    for (x, y), prob in zip(positions, probs):
        fake_p = float(prob[0])
        color = "red" if fake_p > 0.5 else "green"
        # alpha scales with confidence: 0.5 → 0.3, 1.0 → 0.7
        alpha = 0.3 + 0.4 * abs(fake_p - 0.5) * 2
        axes[1].add_patch(
            mpatches.Rectangle(
                (x, y), patch_size, patch_size,
                linewidth=0.4, edgecolor=color, facecolor=color, alpha=alpha,
            )
        )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf).copy()


def run(image: Image.Image, model_name: str) -> tuple[Image.Image | None, str]:
    if image is None:
        return None, "Upload an image first."

    model = load_model(model_name)
    patches, positions, w, h = tile_image(image, patch_size=32)

    if not patches:
        return None, f"Image too small needs at least 32×32 px (got {w}×{h})."

    probs = predict_patches(model, patches)
    fake_p = probs[:, 0]
    mean_fake = float(fake_p.mean())
    mean_real = 1.0 - mean_fake
    verdict = "FAKE" if mean_fake > 0.5 else "REAL"
    confidence = max(mean_fake, mean_real) * 100
    fake_pct = float((fake_p > 0.5).mean()) * 100
    cols = w // 32
    rows = h // 32

    summary = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  Verdict    : {verdict}\n"
        f"  Confidence : {confidence:.1f}%\n"
        f"  FAKE score : {mean_fake:.4f}   REAL score : {mean_real:.4f}\n"
        f"  FAKE patches : {fake_pct:.1f}%  ({cols}×{rows} grid, {len(patches)} patches)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"⚠ Domain shift: trained on 32×32 CIFAKE (CIFAR-10 real + SD v1.4 fake).\n"
        f"  Full-res or other-generator images may yield different accuracy.\n"
        f"  Treat as signal, not ground truth."
    )

    heatmap = render_heatmap(image, positions, probs, patch_size=32)
    return heatmap, summary



DESCRIPTION = """
**Binary classifier**: REAL vs AI-generated (FAKE) images.

**How it works**: tiles your image into 32×32 patches → classifies each patch →
aggregates into a final verdict. The patch map shows where the model suspects
AI generation (red) vs authentic content (green).

**Models trained on**: [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
 120k images (CIFAR-10 real + Stable Diffusion v1.4 fake).
"""

with gr.Blocks(title="SynthSentinel AI Image Detector") as demo:
    gr.Markdown("# SynthSentinel AI Image Detector")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload image (any resolution)")
            model_radio = gr.Radio(
                choices=list(CHECKPOINTS.keys()),
                value="ResNet50 (97.8% acc)",
                label="Model",
            )
            run_btn = gr.Button("Detect", variant="primary")

        with gr.Column(scale=2):
            heatmap_out = gr.Image(type="pil", label="Patch heatmap")
            result_out = gr.Textbox(label="Results", lines=10, max_lines=12)

    run_btn.click(fn=run, inputs=[img_input, model_radio], outputs=[heatmap_out, result_out])
    img_input.change(fn=run, inputs=[img_input, model_radio], outputs=[heatmap_out, result_out])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
