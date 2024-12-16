from ..custom_patch_extraction.dataset import BOPDataset
from ..data.obj import load_objs
from .. import utils
from pathlib import Path
import json
import numpy as np
import cv2

def render_R_t(image, surface_samples, cam_K, obj_id, R, t):
    """
    Projects the object coordinates of a given ovject into the image plane defined by image
    Args:
        image: np.Array() - the image containing the objects
        cam_K: np.Array() - camera matrix
        obj_id: int - Object id
        R, t: np.Array() - pose of the object
    """
    render = image.copy()

    # Create the pose matrix
    pose = np.concatenate((R, t), axis=1)
    # Calculate Projection Matrix
    P = cam_K @ pose

    for surface_sample in surface_samples[obj_id]:
        # Project the surface sample
        surface_sample = np.concatenate((surface_sample, [1]))
        projected = P @ surface_sample
        projected = projected / projected[2]
        x, y = int(projected[0]), int(projected[1])
        # Ensure the projected point is within image bounds
        if 0 <= y < render.shape[0] and 0 <= x < render.shape[1]:
            # Scale green channel in the result image
            render[y, x, :] = render[y, x, :] * 1.5 * np.array([0, 1, 0])

    return render

def render_from_file(image, results_path, render_dir):
    """
    Projects the object coordinates of a given ovject into the image plane defined by image
    Args:
        image: np.Array() - the image containing the objects
        results_path: str - path to the corresponding results_file (.json)
        render_dir: str - path to the directory the render should be saved to
    """
    render = image.copy()

    # Read the results
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Load the objects
    objs, obj_ids = load_objs(Path('./data/bop/itodd/models'))
    surface_samples, surface_sample_normals = utils.load_surface_samples('itodd', obj_ids)

    cam_K = np.array(results['cam_K'])

    for result in results['objects']:
        obj_id = result['obj_id']
        R = np.array(result['R'])
        t = np.array(result['t']).reshape(3, 1)

        # Create the pose matrix
        pose = np.concatenate((R, t), axis=1)
        # Calculate Projection Matrix
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
                    if 0 <= y < render.shape[0] and 0 <= x < render.shape[1]:
                        # Scale green channel in the result image
                        render[y, x, :] = render[y, x, :] * 1.5 * np.array([0, 1, 0])

    return render

if __name__ == '__main__':
    # Load the corresponding image
    dataset = BOPDataset('./data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)
    image, _, _ = dataset[1]
    image = image.permute(1, 2, 0).numpy() * 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    render = render_from_file(image, './results/image_1_results.json', './results')
    cv2.imwrite('./results/image_1_render.png', render)