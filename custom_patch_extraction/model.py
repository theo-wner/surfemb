import pytorch_lightning as pl
import torch
from utils import get_model
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import JaccardIndex

class MaskRCNN(pl.LightningModule):
    def __init__(self, num_classes, learning_rate):
        super().__init__()
        self.model = get_model(num_classes)
        self.learning_rate = learning_rate

        # Metrics for bounding boxes
        self.map_metric = MeanAveragePrecision()

        # Metric for masks
        self.mask_iou = JaccardIndex(task="multilabel", num_labels=1)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return losses
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch

        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("val_loss", losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Compute the mean average precision
        self.map_metric.update(targets, self.model(images))
        return losses

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0005
        )

