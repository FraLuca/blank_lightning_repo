import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class Learner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.model = init_model(cfg, self.model)
        print(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters(cfg)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        accuracy = (y_hat.argmax(1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        accuracy = (y_hat.argmax(1) == y).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def train_dataloader(self):
        train_set = build_dataset(self.cfg, train=True)

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.SOLVER.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,)
        return train_loader

    def val_dataloader(self):
        val_set = build_dataset(self.cfg, train=False)
        
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE_VAL,
            shuffle=False,
            num_workers=self.cfg.SOLVER.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,)
        return val_loader


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=cfg.SOLVER.MOMENTUM)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONE, gamma=0.2)

        return [optimizer], [scheduler]