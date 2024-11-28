import warnings
warnings.showwarning = lambda *args, **kwargs: None

import pytorch_lightning as pl
import torch
from custom_patch_extraction.utils import get_model
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class MaskRCNN(pl.LightningModule):
    def __init__(self, num_classes, learning_rate):
        super().__init__()
        self.model = get_model(num_classes)
        self.learning_rate = learning_rate

        # Metrics for bounding boxes
        self.box_map = MeanAveragePrecision()

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return losses
    
    def on_validation_epoch_start(self):
        self.box_map.reset()  # Reset the metric object

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        self.box_map.update(preds, targets)

    def validation_epoch_end(self, outputs):
        metrics = self.box_map.compute()
        self.log("map_50", metrics['map_50'], on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0005
        )
