from dataset import BOPDataset
from torch.utils.data import DataLoader
from model import MaskRCNN
import pytorch_lightning as pl
import config
from pytorch_lightning.loggers import TensorBoardLogger

# Initialize the dataset
train_dataset = BOPDataset(root_dir='../data/bop/tless', subset='train_pbr')
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.collate_fn)

# Initialize the model
model = MaskRCNN(num_classes=train_dataset.num_classes)

# Initialize the TensorBoard logger
logger = TensorBoardLogger("tb_logs", name="mask_rcnn")

# Move the model to the device
model.to(config.device)

# Initialize the PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu",  # Use GPU
    devices=[1],  # Specify GPU index
    precision=16,  # Mixed precision
    logger=logger,
)

trainer.fit(model, train_dataloader)