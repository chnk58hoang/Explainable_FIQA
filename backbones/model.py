from backbones.iresnet import iresnet50
import torch.nn as nn
import torch


class Explainable_FIQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = iresnet50()
        self.backbone.load_state_dict(torch.load('/kaggle/input/ex-fiqa-code/MYFIQA/pretrained/backbone.pth', map_location='cuda'))
        self.sharpness = nn.Linear(512, 1)
        self.illumination = nn.Linear(512, 1)

        print("Freezing backbone's parameters...")

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        feature = self.backbone(x)
        sharpness = self.sharpness(feature)
        illu = self.illumination(feature)
        illu = illu.squeeze(1)
        sharpness = sharpness.squeeze(1)
        return feature, sharpness, illu
