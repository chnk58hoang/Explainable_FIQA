from backbones.iresnet import iresnet100
import torch.nn as nn
import torch


class Explainable_FIQA(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = iresnet100()
        self.backbone.load_state_dict(torch.load('/kaggle/input/ex-fiqa-code/181952backbone.pth', map_location='cuda'))
        self.sharpness = nn.Linear(1, 1)
        self.illumination = nn.Linear(1, 1)


        print("Freezing backbone's parameters...")

        for param in self.backbone.parameters():
            param.requires_grad = False
    def forward(self, x):
        feature,qs = self.backbone(x)
        sharpness = self.sharpness(qs/2-0.35)
        illu = self.illumination(qs/2-0.35)
        illu = illu.squeeze(1)
        sharpness = sharpness.squeeze(1)
        return feature, sharpness, illu
