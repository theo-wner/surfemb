from .model import MaskRCNN
from .dataset import BOPDataset
from . import config
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Initialize the datasets
train_dataset = BOPDataset('./data/bop/itodd', subset='train_pbr', split='train', test_ratio=0.1)
train_dataloader = DataLoader(train_dataset, 
                              batch_size=config.BATCH_SIZE, 
                              num_workers=config.NUM_WORKERS, 
                              shuffle=True, 
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True,
                              prefetch_factor=2)
val_dataset = BOPDataset('./data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)  
val_dataloader = DataLoader(val_dataset, 
                            batch_size=config.BATCH_SIZE, 
                            num_workers=config.NUM_WORKERS, 
                            shuffle=False,
                            collate_fn=val_dataset.collate_fn, 
                            pin_memory=True,
                            prefetch_factor=2)

# Initialize the model
model = MaskRCNN(num_classes=train_dataset.num_classes, learning_rate=config.LEARNING_RATE)

# Initialize the TensorBoard logger
logger = TensorBoardLogger("tb_logs", name="mask_rcnn")

# Move the model to the device
model.to(config.DEVICE)

# Initialize the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="map_50",    # Save the model with the highest mAP
    save_top_k=1,        
    mode="max",          # For metrics like loss (minimize), use "max" for accuracy
    dirpath="checkpoints",  
    filename="best-checkpoint",  
)

# Initialize the PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu",  # Use GPU
    devices=[config.DEVICE_NUM],  # Specify GPU index
    precision=16,  # Mixed precision
    logger=logger,
    callbacks=[checkpoint_callback]
)

trainer.fit(model, 
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader)