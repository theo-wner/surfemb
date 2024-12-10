from ..custom_patch_extraction.dataset import BOPDataset
from ..custom_patch_extraction.model import MaskRCNN
from ..custom_patch_extraction.utils import infer_detector, visualize_data
from ..surface_embedding import SurfaceEmbeddingModel
from ..data.renderer import ObjCoordRenderer
from ..data.obj import load_objs
from ..pose_est import estimate_pose
from ..pose_refine import refine_pose
from .. import utils
from . import config
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Create Dataset and get image
dataset = BOPDataset('./data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)
image, target, cam_K = dataset[1]
if config.GRAYSCALE:
    image = image.mean(dim=0, keepdim=True)

'''
# Visualize the data
if config.GRAYSCALE:
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
else:
    plt.imshow(image.permute(1, 2, 0))
for box in target['boxes']:
    plt.plot([box[0], box[2], box[2], box[0], box[0]], [box[1], box[1], box[3], box[3], box[1]], 'g')
plt.show()
'''

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

# Create the renderer
res_crop = 224
renderer = ObjCoordRenderer(objs, res_crop)

# Infer the detection model on the image
preds = infer_detector(detection_model, image)

'''
# Visualize the detections
if config.GRAYSCALE:
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
else:
    plt.imshow(image.permute(1, 2, 0))
for box in preds['boxes']:
    plt.plot([box[0], box[2], box[2], box[0], box[0]], [box[1], box[1], box[3], box[3], box[1]], 'r')
plt.show()
'''

# Iterate over the Image crops
for i in range(len(preds['labels'])):
    # Take out the first image crop
    box = preds['boxes'][i]
    obj_idx = preds['labels'][i].item() - 1 # 0-indexed
    # Crop the image
    image_crop = image[:, box[1]:box[3], box[0]:box[2]]
    old_width, old_height = image_crop.shape[1], image_crop.shape[2]
    # Scale image crop to 224x224
    image_crop = torch.nn.functional.interpolate(image_crop.unsqueeze(0), size=(res_crop, res_crop), mode='bilinear', align_corners=False).squeeze(0)
    image_crop = image_crop.permute(1, 2, 0).cpu().numpy()
    image_crop = (image_crop * 255).astype(np.uint8) # Convert to uint8
    # Get scale factor
    scale_x = res_crop / old_width
    scale_y = res_crop / old_height
    assert scale_x == scale_y
    scale = scale_x
    # Correct the camera matrix --> apperently not needed
    K_crop = np.copy(cam_K)

    '''
    # Visualize the data
    if config.GRAYSCALE:
        plt.imshow(image_crop, cmap='gray')
    else:
        plt.imshow(image_crop)
    plt.show()
    '''
    
    # Infer the embedding model on the image crop
    mask_lgts, query_img = embedding_model.infer_cnn(image_crop, obj_idx, rotation_ensemble=False)

    # Get the object and surface samples for the first object
    obj = objs[obj_idx]
    verts = surface_samples[obj_idx]
    verts_norm = (verts - obj.offset) / obj.scale

    # Infer the MLP on the surface samples
    obj_keys = embedding_model.infer_mlp(torch.from_numpy(verts_norm).float(), obj_idx)
    verts = torch.from_numpy(verts).float()

    # Estimate the pose
    R_est, t_est, scores, *_ = estimate_pose(
        mask_lgts=mask_lgts, query_img=query_img,
        obj_pts=verts, obj_normals=surface_sample_normals[obj_idx], obj_keys=obj_keys,
        obj_diameter=obj.diameter, K=K_crop, max_pool=False, visualize=False,
    )
    success = len(scores) > 0
    if success:
        best_idx = torch.argmax(scores).item()
        best_score = scores[best_idx].item()
        R_est, t_est = R_est[best_idx].cpu().numpy(), t_est[best_idx].cpu().numpy()[:, None]
    else:
        print('Pose estimation failed')
        R_est, t_est = np.eye(3), np.zeros((3, 1))

    # Refine the pose
    if success:
        R_est_r, t_est_r, score_r = refine_pose(
            R=R_est, t=t_est, query_img=query_img, K_crop=K_crop,
            renderer=renderer, obj_idx=obj_idx, obj_=obj, model=embedding_model, keys_verts=obj_keys,
        )
    else:
        R_est_r, t_est_r = R_est, t_est

    # Print the results
    print(f'Target R:\n{target["R_t"][0]["R"]}')
    print(f'Target t:\n{target["R_t"][0]["t"]}')
    print(f'Estimated R:\n{R_est}')
    print(f'Estimated t:\n{t_est}')

    
