from ..custom_patch_extraction.dataset import BOPDataset
from ..custom_patch_extraction.model import MaskRCNN
from ..custom_patch_extraction import config
from ..custom_patch_extraction.utils import infer_detector, visualize_data
from ..surface_embedding import SurfaceEmbeddingModel
from ..data.renderer import ObjCoordRenderer
from ..data.obj import load_objs
from ..pose_est import estimate_pose
from ..pose_refine import refine_pose
from .. import utils
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Create Dataset and get image
dataset = BOPDataset('./data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)
image, target, cam_K = dataset[1]

# Visualize the data
plt.imshow(image.permute(1, 2, 0))
for box in target['boxes']:
    plt.plot([box[0], box[2], box[2], box[0], box[0]], [box[1], box[1], box[3], box[3], box[1]], 'g')
plt.show()

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

# Iterate over the Image crops
for i in range(len(preds['labels'])):
    # Take out the first image crop
    box = preds['boxes'][i]
    # Update the camera matrix (Subtract the box coordinates)
    cam_crop = cam_K
    cam_crop[0, 2] -= box[0] # cx
    cam_crop[1, 2] -= box[1] # cy
    # Crop the image
    image_crop = image[:, box[1]:box[3], box[0]:box[2]]
    # Scale image crop to 224x224
    image_crop = torch.nn.functional.interpolate(image_crop.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
    # Correct the camera matrix
    old_width, old_height = box[2] - box[0], box[3] - box[1]
    cam_crop[0, 0] *= 224 / old_width # fx
    cam_crop[1, 1] *= 224 / old_height # fy
    cam_crop[0, 2] *= 224 / old_width # cx
    cam_crop[1, 2] *= 224 / old_height # cy

    obj_idx = preds['labels'][i].item()
    print(f'Object index: {obj_idx}')

    # Visualize the data
    plt.imshow(image_crop.permute(1, 2, 0))
    plt.show()

    # Create the renderer
    renderer = ObjCoordRenderer(objs, w=image_crop.shape[2], h=image_crop.shape[1])

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
        obj_diameter=obj.diameter, K=cam_crop, max_pool=False, visualize=False,
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
            R=R_est, t=t_est, query_img=query_img, K_crop=cam_crop,
            renderer=renderer, obj_idx=obj_idx, obj_=obj, model=embedding_model, keys_verts=obj_keys,
        )
    else:
        R_est_r, t_est_r = R_est, t_est

    # Print the results
    print(f'Target R:\n{target["R_t"][0]["R"]}')
    print(f'Target t:\n{target["R_t"][0]["t"]}')
    print(f'Estimated R:\n{R_est}')
    print(f'Estimated t:\n{t_est}')

    
