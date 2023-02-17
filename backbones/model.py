from backbones.iresnet import iresnet50
import torch.nn as nn
import torch


class Explainable_FIQA(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = iresnet50()
        self.backbone.load_state_dict(torch.load('/kaggle/input/ex-fiqa-code/backbone.pth', map_location='cuda'))

        self.medium = nn.Sequential(*[nn.LazyLinear(256), nn.LazyLinear(64)])

        self.sharpness = nn.Sequential(*[nn.LazyLinear(32), nn.LazyLinear(16), nn.LazyLinear(1)])
        self.illumination = nn.Sequential(*[nn.LazyLinear(32), nn.LazyLinear(16), nn.LazyLinear(1)])
        self.quality = nn.Sequential(*[nn.LazyLinear(32), nn.LazyLinear(16), nn.LazyLinear(1)])

        print("Freezing backbone's parameters...")

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        feature = self.backbone(x)
        medium = self.medium(feature)
        sharpness = self.sharpness(medium)
        illu = self.illumination(medium)
        qscore = self.quality(medium)
        illu = illu.squeeze(1)
        sharpness = sharpness.squeeze(1)
        qscore = qscore.squeeze(1)
        return feature, sharpness, illu, qscore
