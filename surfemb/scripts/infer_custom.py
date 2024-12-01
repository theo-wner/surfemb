from ..custom_patch_extraction.dataset import BOPDataset
from ..custom_patch_extraction.model import MaskRCNN
from ..custom_patch_extraction import config
from ..custom_patch_extraction.utils import infer_detector
from ..surface_embedding import SurfaceEmbeddingModel
from ..data.obj import load_objs
from ..pose_est import estimate_pose
from .. import utils
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Create Dataset and get image
dataset = BOPDataset('./data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)
image, target, cam_K = dataset[10]

# Load the models
detection_model = MaskRCNN.load_from_checkpoint(
    './surfemb/custom_patch_extraction/checkpoints/best-checkpoint.ckpt',
    num_classes=dataset.num_classes,
    learning_rate=config.LEARNING_RATE
)

embedding_model = SurfaceEmbeddingModel.load_from_checkpoint(
    './data/models/itodd-3qnq15p6.compact.ckpt'
).eval()
embedding_model.freeze()

# Get the objects and surface samples
objs, obj_ids = load_objs(Path('./data/bop/itodd/models'))
surface_samples, surface_sample_normals = utils.load_surface_samples('itodd', obj_ids)

# Infer the detection model on the image
preds = infer_detector(detection_model, image)

# Take out the first image crop
first_box = preds['boxes'][1]
first_image_crop = image[:, first_box[1]:first_box[3], first_box[0]:first_box[2]]
first_obj_idx = preds['labels'][1].item()

# Visualize the data
plt.imshow(first_image_crop.permute(1, 2, 0))
plt.show()

# Infer the embedding model on the image crop
mask_lgts, query_img = embedding_model.infer_cnn(first_image_crop, first_obj_idx, rotation_ensemble=False)

# Get the object and surface samples for the first object
obj = objs[first_obj_idx]
verts = surface_samples[first_obj_idx]
verts_norm = (verts - obj.offset) / obj.scale

# Infer the MLP on the surface samples
obj_keys = embedding_model.infer_mlp(torch.from_numpy(verts_norm).float(), first_obj_idx)
verts = torch.from_numpy(verts).float()

# Estimate the pose
R_est, t_est, scores, *_ = estimate_pose(
    mask_lgts=mask_lgts, query_img=query_img,
    obj_pts=verts, obj_normals=surface_sample_normals[first_obj_idx], obj_keys=obj_keys,
    obj_diameter=obj.diameter, K=cam_K,
)
success = len(scores) > 0
if success:
    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()
    R_est, t_est = R_est[best_idx].cpu().numpy(), t_est[best_idx].cpu().numpy()[:, None]
else:
    print('Pose estimation failed')
    R_est, t_est = np.eye(3), np.zeros((3, 1))

print(f'Best score: {best_score}')
print(f'Rotation matrix: {R_est}')
print(f'Translation vector: {t_est}')
