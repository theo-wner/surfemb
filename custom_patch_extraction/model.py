import pytorch_lightning as pl
import torch
from utils import get_model

class MaskRCNN(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001):
        super().__init__()
        self.model = get_model(num_classes)
        self.learning_rate = learning_rate

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses)
        return losses

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0005
        )
