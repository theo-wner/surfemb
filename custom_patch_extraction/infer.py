import warnings
warnings.showwarning = lambda *args, **kwargs: None

from dataset import BOPDataset
from model import MaskRCNN
import config
from utils import visualize_data, infer


# Create Dataset and get image
dataset = BOPDataset('../data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)
image, target = dataset[1000]

# Load the model
model = MaskRCNN.load_from_checkpoint(
    './checkpoints/best-checkpoint.ckpt',
    num_classes=dataset.num_classes,
    learning_rate=config.LEARNING_RATE
)

# Infer the model on the image
preds = infer(model, image)

# Visualize the data
visualize_data(image, target, preds)