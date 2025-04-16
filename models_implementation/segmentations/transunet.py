from torch import nn
from ..classifications.vitb16 import ViT
import torch

class StemBlock(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)

        return x

class TransUNetEncoder(nn.Module):
    def __init__(self, img_size: int, embed_dim: int = 768, in_channels: int = 3, base_channels: int = 64, depth=12):
        super().__init__()

        self.stem_block1 = StemBlock(in_channels, base_channels)
        self.stem_block2 = StemBlock(base_channels, base_channels*2)
        self.stem_block3 = StemBlock(base_channels*2, base_channels*4)

        self.linear_proj = nn.Conv2d(4*base_channels, embed_dim, kernel_size=1)
        self.vit = ViT(img_size//8, in_channels=embed_dim,
                       embed_dim=embed_dim,
                       dropout=0.1, num_classes=1, patch_size=1, depth=depth)

        self.embed_dim = embed_dim
        self.gh = self.gw = img_size//8

    def forward(self, x):
        x1 = self.stem_block1(x)
        x2 = self.stem_block2(x1)
        x3 = self.stem_block3(x2)

        y = self.linear_proj(x3)
        features = self.vit.forward_features(y)[:,1:]

        t2 = features.transpose(1,2)
        t2 = t2.contiguous().view(-1,
                          self.embed_dim,
                          self.gh,
                          self.gw)

        return x1, x2, x3, t2

class TransUnetDecoder(nn.Module):
    def __init__(self,
                 base_channels: int = 64,
                 embed_dim:     int = 768,
                 num_classes:   int = 1):
        super().__init__()
        C1, C2, C3 = base_channels, base_channels*2, base_channels*4

        self.t2_conv = nn.Sequential(
            nn.Conv2d(embed_dim,      C3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C3),
            nn.ReLU(inplace=True),
            nn.Conv2d(C3,             C3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C3),
            nn.ReLU(inplace=True),
        )

        self.bridge_conv = nn.Sequential(
            nn.Conv2d(C3*2, C3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C3),
            nn.ReLU(inplace=True),
            nn.Conv2d(C3,   C3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C3),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(C3, C2, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(C2*2, C2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C2,   C2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C2),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(C2, C1, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(C1*2, C1, 3, padding=1, bias=False),
            nn.BatchNorm2d(C1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C1,   C1, 3, padding=1, bias=False),
            nn.BatchNorm2d(C1),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(C1, C1, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(C1, C1, 3, padding=1, bias=False),
            nn.BatchNorm2d(C1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C1, C1, 3, padding=1, bias=False),
            nn.BatchNorm2d(C1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(C1, num_classes, kernel_size=1)

    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor,
                x3: torch.Tensor,
                t2: torch.Tensor
               ) -> torch.Tensor:
        # 1) Предварительная обработка t2
        t2p = self.t2_conv(t2)

        # 2) Bridge: объединяем x3 и t2p
        b = torch.cat([x3, t2p], dim=1)
        b = self.bridge_conv(b)

        # 3) Level 1 decode
        d1 = self.up1(b)
        d1 = torch.cat([d1, x2], dim=1)
        d1 = self.dec1(d1)

        # 4) Level 2 decode
        d2 = self.up2(d1)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.dec2(d2)

        # 5) Level 3 decode
        d3 = self.up3(d2)
        d3 = self.dec3(d3)

        # 6) Segmentation head
        return self.head(d3)

class TransUNet(nn.Module):
    def __init__(self, img_size: int, num_classes: int = 1, in_channels: int = 3, embed_dim: int = 768, base_channels: int = 64, depth=12):
        super().__init__()
        self.encoder = TransUNetEncoder(img_size, embed_dim, in_channels, base_channels, depth)
        self.decoder = TransUnetDecoder(base_channels, embed_dim, num_classes)

    def forward(self, x):
        x1, x2, x3, t2 = self.encoder(x)
        x = self.decoder(x1, x2, x3, t2)
        return x
