import pytorch_lightning as pl
from tqdm import tqdm
from torch.nn.functional import l1_loss


class MyCallBack(pl.Callback):
    def __init__(self, val_loader, test_loader):
        super().__init__()
        self.test_loader = test_loader
        self.val_loader = val_loader

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        sharp_mae = 0.0
        illu_mae = 0.0
        q_mae = 0.0
        count = 0
        for batch_idx, batch in tqdm(enumerate(self.val_loader)):
            count += 1
            image, sharpness, illu, qscore = batch
            sharpness = sharpness.to(pl_module._device)
            illu = illu.to(pl_module._device)
            qscore = qscore.to(pl_module._device)
            _, pred_sharp, pred_illu, pred_q = pl_module(image, sharpness, illu, qscore)

            sharp_mae += l1_loss(sharpness, pred_sharp, reduction='mean')
            illu_mae += l1_loss(illu, pred_illu, reduction='mean')
            q_mae += l1_loss(qscore, pred_q, reduction='mean')

        sharp_mae /= count
        illu_mae /= count
        q_mae /= count

        print('MAE for sharpness {}'.format(float(sharp_mae)))
        print('MAE for illumination {}'.format(float(illu_mae)))
        print('MAE for qscore {}'.format(float(q_mae)))

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        sharp_mae = 0.0
        illu_mae = 0.0
        q_mae = 0.0
        count = 0
        for batch_idx, batch in tqdm(enumerate(self.test_loader)):
            count += 1
            image, sharpness, illu, qscore = batch
            sharpness = sharpness.to(pl_module._device)
            illu = illu.to(pl_module._device)
            qscore = qscore.to(pl_module._device)
            _, pred_sharp, pred_illu, pred_q = pl_module(image, sharpness, illu, qscore)

            sharp_mae += l1_loss(sharpness, pred_sharp, reduction='mean')
            illu_mae += l1_loss(illu, pred_illu, reduction='mean')
            q_mae += l1_loss(qscore, pred_q, reduction='mean')

        sharp_mae /= count
        illu_mae /= count
        q_mae /= count

        print('MAE for sharpness {}'.format(float(sharp_mae)))
        print('MAE for illumination {}'.format(float(illu_mae)))
        print('MAE for qscore {}'.format(float(q_mae)))
