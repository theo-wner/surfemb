from dataset import BOPDataset
from torch.utils.data import DataLoader
from model import MaskRCNN
import pytorch_lightning as pl
import config
from pytorch_lightning.loggers import TensorBoardLogger

# Initialize the datasets
train_dataset = BOPDataset('../data/bop/itodd', subset='train_pbr', split='train', test_ratio=0.1)
train_dataloader = DataLoader(train_dataset, 
                              batch_size=config.batch_size, 
                              num_workers=config.num_workers, 
                              shuffle=True, 
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True,
                              prefetch_factor=4)
val_dataset = BOPDataset('../data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)  
val_dataloader = DataLoader(val_dataset, 
                            batch_size=config.batch_size, 
                            num_workers=config.num_workers, 
                            shuffle=False,
                            collate_fn=val_dataset.collate_fn)
                            #pin_memory=True,
                            #prefetch_factor=2)

# Initialize the model
model = MaskRCNN(num_classes=train_dataset.num_classes, learning_rate=config.learning_rate)

# Initialize the TensorBoard logger
logger = TensorBoardLogger("tb_logs", name="mask_rcnn")

# Move the model to the device
model.to(config.device)

# Initialize the PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu",  # Use GPU
    devices=[config.device_num],  # Specify GPU index
    precision=16,  # Mixed precision
    logger=logger
    #val_check_interval=1,  # Check validation every epoch
)

trainer.fit(model, 
            train_dataloader=train_dataloader, 
            val_dataloaders=val_dataloader)