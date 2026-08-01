import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class BasicBlock(nn.Module):
    """Standard ResNet-18/34 residual block (two 3x3 convs + identity/1x1
    shortcut). Used as the building block for CIFARResNetEncoder below."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CIFARResNetEncoder(nn.Module):
    """CIFAR-style ResNet-18 backbone (3x3 stride-1 stem instead of
    ImageNet's 7x7 stride-2 + maxpool, which would crush a 32x32 input) plus
    a small SSL projector head. `forward` drives LeJEPA pretraining over
    multiple views; `embed` exposes the raw 512-dim feature a classifier
    head can attach to at fine-tune time."""

    def __init__(self, embedding_dim=512, proj_dim=64):
        super().__init__()
        assert embedding_dim == 512, "embedding_dim is fixed by ResNet-18's layer4 width"

        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 3-layer projector (512 -> 2048 -> 2048 -> proj_dim) mapping the raw
        # backbone embedding into the space SIGReg and the JEPA prediction
        # loss actually operate on. 
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, proj_dim),
        )

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def embed(self, x):
        """Raw 512-dim backbone embedding (b c h w) bypassing the SSL projector."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = rearrange(x, 'b c 1 1 -> b c')  # drop the trailing 1x1 spatial dims
        return x

    def forward(self, x):
        """SSL forward pass: x is (n v c h w), all n*v views embedded through
        the shared encoder then projected. Returns emb (n*v, embedding_dim)
        and proj (v, n, proj_dim), views-first for the LeJEPA loss."""
        n, v = x.shape[:2]
        x = rearrange(x, 'n v c h w -> (n v) c h w')
        emb = self.embed(x)
        proj = self.projector(emb)
        proj = rearrange(proj, '(n v) d -> v n d', n=n, v=v)
        return emb, proj


class LeJEPAClassifier(nn.Module):
    """Classifier head for fine-tuning a (pretrained) CIFARResNetEncoder.

    Replaces the SSL projector with a single linear layer over `embed()`'s
    512-dim backbone output the projector is SSL-only scaffolding &
    plays no role at fine-tune time
    """

    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.head(self.encoder.embed(x))