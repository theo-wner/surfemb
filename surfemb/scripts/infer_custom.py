from custom_patch_extraction.dataset import BOPDataset
from custom_patch_extraction.model import MaskRCNN
from custom_patch_extraction import config
from custom_patch_extraction.utils import infer_detector, visualize_data

# Create Dataset and get image
dataset = BOPDataset('../data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)
image, target = dataset[1000]

# Load the model
model = MaskRCNN.load_from_checkpoint(
    './custom_patch_extraction/checkpoints/best-checkpoint.ckpt',
    num_classes=dataset.num_classes,
    learning_rate=config.LEARNING_RATE
)

# Infer the model on the image
preds = infer_detector(model, image)

# Take out the first image crop and infer on it
first_box = preds['boxes'][0]
first_image_crop = image[:, first_box[1]:first_box[3], first_box[0]:first_box[2]]

