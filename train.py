from backbones.model import Explainable_FIQA
from dataset.dataset import ExFIQA
from torch.utils.data import DataLoader
from callback.callback import MyCallBack
import pytorch_lightning as pl
import torch.nn as nn
import pandas as pd
import torch


class EXFIQA(pl.LightningModule):
    def __init__(self, model, train_loader, val_loader, device):
        super().__init__()
        self.model = model
        self._device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss1 = nn.L1Loss()
        self.loss2 = nn.L1Loss()
        self.loss3 = nn.L1Loss()
        self.automatic_optimization = False

    def forward(self, image, sharp, illu, qscore):
        image = image.to(self._device)
        sharp = sharp.to(self._device)
        illu = illu.to(self._device)
        qscore = qscore.to(self._device)

        return self.model(image)

    def configure_optimizers(self):
        optim1 = torch.optim.Adam(self.model.sharpness.parameters(), lr=1e-2)
        optim2 = torch.optim.Adam(self.model.illumination.parameters(), lr=1e-2)
        optim3 = torch.optim.Adam(self.model.quality.parameters(), lr=1e-2)

        sch1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim1, factor=0.5, patience=2)
        sch2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim2, factor=0.5, patience=2)
        sch3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim3, factor=0.5, patience=2)

        return [optim1, optim2, optim3], [sch1, sch2, sch3]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_idx):
        (opt1, opt2, opt3) = self.optimizers()
        (sch1, sch2, sch3) = self.lr_schedulers()

        image, sharp, illu, qscore = batch
        sharp = sharp.to(self._device)
        illu = illu.to(self._device)
        qscore = qscore.to(self._device)
        _, pred_sharp, pred_illu, pred_qscore = self.forward(image, sharp, illu, qscore)

        loss3 = self.loss3(qscore, pred_qscore)
        self.manual_backward(loss3)
        opt3.step()
        sch3.step(loss3)

        return {'loss': loss3}

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self):
        print('Start testing...')


if __name__ == '__main__':
    dataframe = pd.read_csv('/kaggle/input/ex-fiqa-code/newdata2.csv')
    dataframe = dataframe.sample(frac=1)
    train_len = int(len(dataframe) * 0.8)
    valid_len = int(len(dataframe) * 0.9)
    # test_len = int(len(dataframe) * 0.2)
    train_dataframe = dataframe.iloc[:train_len, :]
    valid_dataframe = dataframe.iloc[train_len:valid_len, :]
    test_dataframe = dataframe.iloc[valid_len:, :]

    model = Explainable_FIQA()
    train_dataset = ExFIQA(df=train_dataframe)
    valid_dataset = ExFIQA(df=valid_dataframe)
    test_dataset = ExFIQA(df=test_dataframe)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

    module = EXFIQA(model=model, train_loader=train_loader, val_loader=val_loader, device=torch.device('cuda'))
    callback = MyCallBack(val_loader, test_loader)

    trainer = pl.Trainer(max_epochs=20, callbacks=[callback, ], auto_lr_find=True, accelerator='gpu')

    trainer.fit(module)
    trainer.test(module, test_loader)

    torch.save(model, 'model.pth')
