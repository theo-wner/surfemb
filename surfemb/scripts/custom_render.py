from ..custom_patch_extraction.dataset import BOPDataset
from ..data.obj import load_objs
from .. import utils
from pathlib import Path
import json
import numpy as np
import cv2

# Load the corresponding image
dataset = BOPDataset('./data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)
image, target, cam_K = dataset[1]

# Save the image in RGB format
rgb = image.permute(1, 2, 0).numpy() * 255
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
cv2.imwrite('image.png', rgb)

# Read the results
with open('results.json', 'r') as f:
    results = json.load(f)

# Load the objects
objs, obj_ids = load_objs(Path('./data/bop/itodd/models'))
surface_samples, surface_sample_normals = utils.load_surface_samples('itodd', obj_ids)

for result in results:
    obj_id = result['obj_id']
    K_crop = np.array(result['K_crop'])
    R = np.array(result['R'])
    t = np.array(result['t']).reshape(3, 1)

    # Create the pose matrix belonging to the cropped image
    pose_crop = np.concatenate((R, t), axis=1)

    # Create the pose matrix belonging to the original image
    # K_crop * (R|t)_crop != K_cam * (R|t)_cam <=> (R|t)_cam = inv(K_cam) * K_crop * (R|t)_crop
    pose = np.linalg.inv(cam_K) @ K_crop @ pose_crop

    # Use the original camera matrix for projection
    P = cam_K @ pose

    for obj in objs:
        if obj.obj_id == obj_id:
            for surface_sample in surface_samples[obj_id - 1]:
                # Project the surface sample
                surface_sample = np.concatenate((surface_sample, [1]))
                projected = P @ surface_sample
                projected = projected / projected[2]
                x, y = int(projected[0]), int(projected[1])

                # Ensure the projected point is within image bounds
                if 0 <= y < image.shape[1] and 0 <= x < image.shape[2]:
                    # Scale each channel in the original image
                    image[:, y, x] = image[:, y, x] * 1.1

    # Save the updated image in RGB format
    rgb = image.permute(1, 2, 0).numpy() * 255
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite('render.png', rgb)