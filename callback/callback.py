import pytorch_lightning as pl
import torch.nn as nn
from tqdm import tqdm


class MyCallBack(pl.Callback):
    def __init__(self, val_loader, test_loader):
        super().__init__()
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.mae = nn.L1Loss()


    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        sharp_mae = 0.0
        illu_mae = 0.0
        num_samples = 0
        for batch_idx, batch in tqdm(enumerate(self.val_loader)):
            image, sharpness, illu = batch
            num_samples += image.size(0)
            _, pred_sharp, pred_illu = pl_module(image)

            sharp_mae += self.mae(sharpness, pred_sharp)
            illu_mae += self.mae(illu, pred_illu)

        sharp_mae /= num_samples
        illu_mae /= num_samples

        print('MAE for sharpness {}'.format(float(sharp_mae)))
        print('MAE for illumination {}'.format(float(illu_mae)))

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        sharp_mae = 0.0
        illu_mae = 0.0
        num_samples = 0
        for batch_idx, batch in enumerate(self.test_loader):
            image, sharpness, illu = batch
            num_samples += image.size(0)
            _, pred_sharp, pred_illu = pl_module(image)

            sharp_mae += self.mae(sharpness, pred_sharp)
            illu_mae += self.mae(illu, pred_illu)

        sharp_mae /= num_samples
        illu_mae /= num_samples

        print('MAE for sharpness {}'.format(float(sharp_mae)))
        print('MAE for illumination {}'.format(float(illu_mae)))

