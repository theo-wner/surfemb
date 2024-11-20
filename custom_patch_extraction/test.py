from model import MaskRCNN
from dataset import BOPDataset
from utils import visualize_data
import config
import torch

train_dataset = BOPDataset(root_dir='../data/bop/tless', split='train_pbr')
test_dataset = BOPDataset(root_dir='../data/bop/tless', split='test_primesense')

# Load the model
model = MaskRCNN.load_from_checkpoint(
    './tb_logs/mask_rcnn/version_0/checkpoints/epoch=9-step=62499.ckpt',
    num_classes=test_dataset.num_classes,
    learning_rate=config.learning_rate
)

# Move the model to the device
model.to(config.device)

# Set the model to evaluation mode
model.eval()

# Test the model
image, target = train_dataset[200]
image = image.unsqueeze(0).to(config.device)

with torch.no_grad():
    preds = model(image)

# Make preds readable for visualization
image = image[0].cpu()
preds = preds[0]
boxes= preds['boxes'].cpu()
labels = preds['labels'].cpu()
masks= preds['masks'].squeeze(dim=1).cpu()
preds = {'boxes': boxes, 'labels': labels, 'masks': masks}

visualize_data(image, target, preds)

    