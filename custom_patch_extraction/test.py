from model import MaskRCNN
from dataset import BOPDataset
from utils import visualize_data, compute_iou
import config
import torch

train_dataset = BOPDataset('../data/bop/itodd', subset='train_pbr', split='train', test_ratio=0.1)
test_dataset = BOPDataset('../data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)  

# Load the model
model = MaskRCNN.load_from_checkpoint(
    './checkpoints/best-checkpoint.ckpt',
    num_classes=test_dataset.num_classes,
    learning_rate=config.learning_rate
)

# Move the model to the device
model.to(config.device)

# Set the model to evaluation mode
model.eval()

# Test the model
image, target = train_dataset[100]
image = image.unsqueeze(0).to(config.device)

with torch.no_grad():
    preds = model(image)

# Make preds readable for visualization
image = image[0].cpu()
preds = preds[0]

# Filter overlapping boxes
iou_threshold = 0.2
keep = torch.ones(len(preds['boxes']), dtype=torch.bool)
for i in range(len(preds['boxes'])):
    if keep[i]:
        for j in range(i + 1, len(preds['boxes'])):
            if keep[j]:
                iou = compute_iou(preds['boxes'][i].cpu().numpy(), preds['boxes'][j].cpu().numpy())
                if iou > iou_threshold:
                    if preds['scores'][i] > preds['scores'][j]:
                        keep[j] = 0
                    else:
                        keep[i] = 0
                        break

# Apply filtering
preds['boxes'] = preds['boxes'][keep]
preds['labels'] = preds['labels'][keep]
preds['scores'] = preds['scores'][keep]
preds['masks'] = preds['masks'][keep]

boxes= preds['boxes'].cpu()
labels = preds['labels'].cpu()
masks= preds['masks'].squeeze(dim=1).cpu()
preds = {'boxes': boxes, 'labels': labels, 'masks': masks}

visualize_data(image, target, preds)

    