from dataset import BOPDataset
from torch.utils.data import DataLoader
from model import MaskRCNN
import pytorch_lightning as pl

# Initialize the dataset
train_dataset = BOPDataset(root_dir='data/tless', split='train_pbr')
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Initialize the model
model = MaskRCNN(num_classes=train_dataset.num_classes)

# Train the model
trainer = pl.Trainer(
    max_epochs=10,
    devices=[1],  # Specify GPU index, e.g., '1' for cuda:1
    accelerator="gpu",  # Use GPU
    precision=16,  # Optional: Mixed precision for faster training
)

trainer.fit(model, train_dataloader)