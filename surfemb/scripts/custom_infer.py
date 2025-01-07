from ..patch_extraction.dataset import BOPDataset
from ..patch_extraction.model import MaskRCNN
from ..patch_extraction.utils import infer_detector
from ..surface_embedding import SurfaceEmbeddingModel
from ..data.renderer import ObjCoordRenderer
from ..data.obj import load_objs
from ..pose_est import estimate_pose
from ..pose_refine import refine_pose
from .. import utils
from . import params_config
from . import custom_render
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import cv2
from scipy.spatial.transform import Rotation  

def infer_surfemb(image, image_name, cam_K, preds, embedding_model, objs, obj_ids, surface_samples, surface_sample_normals, renderer, res_crop):
    """
    Infer the Surface Embedding model on the image crops
    
    Args:
    - image: The original image
    - image_name: The name of the image
    - cam_K: The camera matrix
    - preds: The predictions from the detection model
    - embedding_model: The Surface Embedding model
    - objs: The objects
    - obj_ids: The object ids
    - surface_samples: The surface samples
    - surface_sample_normals: The surface sample normals
    - renderer: The renderer (from the original repo)
    - res_crop: The resolution of the image crops

    Returns:
    - results: The results of the inference (cam_K, R and t for each object) in a dictionary

    Also saves the rendered object points as an image to the results folder for testing purposes
    """

    # Create initial render for testing
    render = image.permute(1, 2, 0).numpy() * 255
    render = cv2.cvtColor(render, cv2.COLOR_BGR2RGB)

    # Create results-file
    results = {'cam_K' : cam_K.tolist(), 'objects' : []}

    # Iterate over the Image crops
    for i in tqdm(range(len(preds['labels'])), desc='infering surfemb on the predictions'):
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
        # Correct the camera matrix
        K_crop = np.copy(cam_K)
        K_crop[0, 0] /= scale # fx
        K_crop[1, 1] /= scale # fy
        K_crop[0, 2] = K_crop[0, 2] / scale + box[0] # cx
        K_crop[1, 2] = K_crop[1, 2] / scale + box[1] # cy
        
        # Infer the embedding model on the image crop
        mask_lgts, query_img = embedding_model.infer_cnn(image_crop, obj_idx, rotation_ensemble=False)

        # Get the object and surface samples for the first object
        obj = objs[obj_idx]
        verts = surface_samples[obj_idx]
        verts_norm = (verts - obj.offset) / obj.scale

        # Infer the MLP on the surface samples
        obj_keys = embedding_model.infer_mlp(torch.from_numpy(verts_norm).float().to(params_config.DEVICE), obj_idx)
        verts = torch.from_numpy(verts).float().to(params_config.DEVICE)

        # Estimate the pose
        R_est, t_est, scores, *_ = estimate_pose(
            mask_lgts=mask_lgts, query_img=query_img,
            obj_pts=verts, obj_normals=surface_sample_normals[obj_idx], obj_keys=obj_keys,
            obj_diameter=obj.diameter, K=cam_K, max_pool=False, visualize=False,
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
                R=R_est, t=t_est, query_img=query_img, K_crop=cam_K,
                renderer=renderer, obj_idx=obj_idx, obj_=obj, model=embedding_model, keys_verts=obj_keys,
            )
        else:
            R_est_r, t_est_r = R_est, t_est

        # Adjust the pose to match the image and not the crop
        pose_crop = np.concatenate((R_est_r, t_est_r), axis=1)
        # Create the pose matrix belonging to the original image
        # K_crop * (R|t)_crop != K_cam * (R|t)_cam <=> (R|t)_cam = inv(K_cam) * K_crop * (R|t)_crop
        pose = np.linalg.inv(cam_K) @ K_crop @ pose_crop
        R = pose[:, :3]
        t = pose[:, 3].reshape(3, 1)
        # Convert R to Euler angles
        r = Rotation.from_matrix(R)
        euler = r.as_euler('zyx', degrees=True)
        # Render the image
        render = custom_render.render_R_t(render, surface_samples, cam_K, obj_idx, R, t)
        cv2.imwrite(f'./results/{image_name}_render.png', render)
        # Append the corrected pose to results
        results['objects'].append({
            'obj_id': obj_ids[obj_idx],
            'R': R.tolist(),
            't': t.tolist(),
            'euler': euler.tolist(),
        })

    return results

if __name__ == '__main__':
    # Create Dataset and get image
    dataset = BOPDataset('./data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)
    image, target, cam_K = dataset[1]
    image_name = 'test_image'
    if params_config.GRAYSCALE:
        image = image.mean(dim=0, keepdim=True)

    # Load the models
    detection_model = MaskRCNN.load_from_checkpoint(
        './surfemb/patch_extraction/checkpoints/best-checkpoint.ckpt',
        num_classes=dataset.num_classes,
        learning_rate=params_config.LEARNING_RATE
    ).to(params_config.DEVICE)

    embedding_model = SurfaceEmbeddingModel.load_from_checkpoint(
        './data/models/itodd-3qnq15p6.compact.ckpt'
    ).eval().to(params_config.DEVICE)
    embedding_model.freeze()

    # Get the objects and surface samples
    objs, obj_ids = load_objs(Path('./data/bop/itodd/models'))
    surface_samples, surface_sample_normals = utils.load_surface_samples('itodd', obj_ids)

    # Create the renderer
    res_crop = 224
    renderer = ObjCoordRenderer(objs, res_crop)

    # Infer the detection model on the image
    preds = infer_detector(detection_model, image)
    print(preds['boxes'])
    print(preds['labels'])

    # Infer the surfemb model on the predictions
    results = infer_surfemb(image, image_name, cam_K, preds, embedding_model, objs, obj_ids, surface_samples, surface_sample_normals, renderer, res_crop)

    with open(f'./results/{image_name}_results.json', 'w') as f:
        json.dump(results, f)

