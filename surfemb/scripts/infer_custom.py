from custom_patch_extraction.dataset import BOPDataset
from custom_patch_extraction.model import MaskRCNN
from ..surface_embedding import SurfaceEmbeddingModel
from custom_patch_extraction import config
from custom_patch_extraction.utils import infer_detector
import matplotlib.pyplot as plt

# Create Dataset and get image
dataset = BOPDataset('../data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)
image, target, cam_K = dataset[10]

# Load the models
detection_model = MaskRCNN.load_from_checkpoint(
    './custom_patch_extraction/checkpoints/best-checkpoint.ckpt',
    num_classes=dataset.num_classes,
    learning_rate=config.LEARNING_RATE
)

embedding_model = SurfaceEmbeddingModel.load_from_checkpoint(
    '../data/models/itodd-3qnq15p6.compact.ckpt'
).eval()
embedding_model.freeze()

# Infer the detection model on the image
preds = infer_detector(detection_model, image)

# Take out the first image crop and infer on it
first_box = preds['boxes'][1]
first_image_crop = image[:, first_box[1]:first_box[3], first_box[0]:first_box[2]]
first_obj_idx = preds['labels'][1].item()

# Infer the embedding model on the first image crop



