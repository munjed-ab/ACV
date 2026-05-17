# SynthSentinel: AI-Generated Image Detection using CNNs

Binary classifier (REAL / FAKE) trained on CIFAKE dataset. Compares custom CNN baseline vs. fine-tuned ResNet50, with Grad-CAM visualization for model interpretability.

---

## Project Spec

| Item | Detail |
|------|--------|
| Dataset | CIFAKE: 120,000 images (60k REAL from CIFAR-10, 60k FAKE from Stable Diffusion v1.4) |
| Task | Binary classification: REAL vs. FAKE |
| Image size | 32×32 RGB, 10 categories |
| Tools | Python, PyTorch, torchvision, scikit-learn, matplotlib |

**Methodology (as submitted to professor):**
1. Data preprocessing and normalization
2. Custom CNN: trained from scratch as baseline
3. ResNet50: fine-tuned via transfer learning
4. Comparison using Accuracy, F1-Score, and Confusion Matrix
5. Grad-CAM: visualize where each model "looks" to make decisions

---

## File Structure

```
project/
 data.py        # dataset loading, transforms, DataLoaders
 models.py      # BaselineCNN + ResNetTransfer
 train.py       # training loop, early stopping, checkpointing
 evaluate.py    # metrics, confusion matrix plot, loss/acc curves
 gradcam.py     # Grad-CAM + overlay visualization
 main.py        # end-to-end pipeline
 data/train/REAL/
 data/train/FAKE/
 data/test/REAL/
 data/test/FAKE/
```

## Results

### Model Comparison

| Model | Accuracy | F1-Score | Epochs |
|-------|----------|----------|--------|
| Baseline CNN | 0.9453 | 0.9452 | 19 (early stop) |
| ResNet50 (Transfer) | **0.9779** | **0.9779** | 27 (early stop) |

Transfer learning improved accuracy by **+3.3%** over the from-scratch baseline.

### Baseline CNN

```
              precision    recall  f1-score   support

        FAKE       0.97      0.92      0.94     10000
        REAL       0.93      0.97      0.95     10000

    accuracy                           0.95     20000
   macro avg       0.95      0.95      0.95     20000
```

- Strong FAKE precision (0.97), low false alarm rate
- Slight asymmetry: FAKE recall 0.92 vs REAL recall 0.97, misses some AI-generated images
- Trained from scratch in 19 epochs, converged fast

### ResNet50 (Transfer Learning)

```
              precision    recall  f1-score   support

        FAKE       0.98      0.98      0.98     10000
        REAL       0.98      0.98      0.98     10000

    accuracy                           0.98     20000
   macro avg       0.98      0.98      0.98     20000
```

- Perfectly balanced, 0.98 precision/recall on both classes
- Fixed the baseline's asymmetry completely
- Backbone frozen for first 5 epochs, unfrozen at lower LR (3e-5) afterward
- Final train loss: 0.026, very confident predictions
- ImageNet texture features transfer well even at 32×32; Stable Diffusion leaves detectable texture artifacts

### Plots

Saved to `results/`:
- `baseline_cnn_curves.png`: loss & accuracy over epochs
- `baseline_cnn_confusion.png`: confusion matrix
- `resnet50_curves.png`: loss & accuracy over epochs
- `resnet50_confusion.png`: confusion matrix
- `gradcam/baseline_cnn_sample_0..4.png`: Grad-CAM overlays
- `gradcam/resnet50_sample_0..4.png`: Grad-CAM overlays

---

## Setup

```bash
pip install torch torchvision scikit-learn matplotlib seaborn tqdm pillow

# Download dataset (requires Kaggle API key at ~/.kaggle/kaggle.json)
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
unzip cifake*.zip -d data/cifake

# Run full pipeline
python main.py
```
