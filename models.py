import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout2d(0.25),
            )

        self.features = nn.Sequential(
            block(3,   32),   # 32→16
            block(32,  64),   # 16→8
            block(64,  128),  # 8→4
            block(128, 256),  # 4→2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResNetTransfer(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # drop original fc
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 2),
        )
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.head(self.backbone(x))
