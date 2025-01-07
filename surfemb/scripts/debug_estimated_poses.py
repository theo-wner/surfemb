from ..patch_extraction.dataset import BOPDataset
from ..patch_extraction.model import MaskRCNN
from ..patch_extraction.utils import calculate_new_bounding_box, visualize_detections, infer_detector
from ..surface_embedding import SurfaceEmbeddingModel
from ..data.renderer import ObjCoordRenderer
from ..data.obj import load_objs
from .. import utils
from . import params_config
from .custom_infer import infer_surfemb
from pathlib import Path
import os
import numpy as np
from io import StringIO
import torch

def target_to_string(target):
    buffer = StringIO()
    
    buffer.write(f'Target\n')
    for i in range(target['boxes'].size(0)):
        label = target['labels'][i].item()
        t = target['R_t'][i]['t']
        euler = target['R_t'][i]['euler']
        buffer.write(f'Object {label}\n')
        buffer.write(f'Translation:\n')
        buffer.write(f'[{t[0][0]:.6f}]\n')  
        buffer.write(f'[{t[1][0]:.6f}]\n')
        buffer.write(f'[{t[2][0]:.6f}]\n')
        buffer.write(f'Euler angles:\n')
        buffer.write(f'[{euler[0]:.6f}]\n')
        buffer.write(f'[{euler[1]:.6f}]\n')
        buffer.write(f'[{euler[2]:.6f}]\n\n')
    
    result = buffer.getvalue()
    buffer.close()
    return result

def results_to_string(results):
    buffer = StringIO()
    
    buffer.write(f'Results\n')
    for object in results['objects']:
        buffer.write(f'Object {object["obj_id"]}\n')
        buffer.write(f'Translation:\n')
        buffer.write(f'[{object["t"][0][0]:.6f}]\n')
        buffer.write(f'[{object["t"][1][0]:.6f}]\n')
        buffer.write(f'[{object["t"][2][0]:.6f}]\n')
        buffer.write(f'Euler angles:\n')
        buffer.write(f'[{object["euler"][0]:.6f}]\n')
        buffer.write(f'[{object["euler"][1]:.6f}]\n')
        buffer.write(f'[{object["euler"][2]:.6f}]\n\n')
    
    result = buffer.getvalue()
    buffer.close()
    return result

def ensemble_results_to_string(results):
    buffer = StringIO()
    
    buffer.write(f'Ensemble Results\n')
    for result in results:
        buffer.write(f'Object {result["obj_id"]}\n')
        buffer.write(f'Translation:\n')
        buffer.write(f'[{result["t_mean"][0]:.6f}] ± [{result["t_std"][0]:.6f}]\n')
        buffer.write(f'[{result["t_mean"][1]:.6f}] ± [{result["t_std"][1]:.6f}]\n')
        buffer.write(f'[{result["t_mean"][2]:.6f}] ± [{result["t_std"][2]:.6f}]\n')
        buffer.write(f'Euler angles:\n')
        buffer.write(f'[{result["euler_mean"][0]:.6f}] ± [{result["euler_std"][0]:.6f}]\n')
        buffer.write(f'[{result["euler_mean"][1]:.6f}] ± [{result["euler_std"][1]:.6f}]\n')
        buffer.write(f'[{result["euler_mean"][2]:.6f}] ± [{result["euler_std"][2]:.6f}]\n\n')

    result = buffer.getvalue()
    buffer.close()
    return result

def print_side_by_side(str1, str2, padding=4):
    """
    Print two multi-line strings side by side.
    
    :param str1: The first multi-line string.
    :param str2: The second multi-line string.
    :param padding: Number of spaces between the two strings.
    """
    lines1 = str1.splitlines()
    lines2 = str2.splitlines()
    
    # Determine the maximum width of the first column
    max_width1 = max(len(line) for line in lines1) if lines1 else 0
    spacer = " " * padding  # Spacer between the columns

    # Make both lists of lines the same length by padding with empty strings
    max_lines = max(len(lines1), len(lines2))
    lines1.extend([""] * (max_lines - len(lines1)))
    lines2.extend([""] * (max_lines - len(lines2)))

    # Print the lines side by side
    for line1, line2 in zip(lines1, lines2):
        print(f"{line1:<{max_width1}}{spacer}{line2}")


if __name__ == '__main__':
    # Create Dataset and get image
    dataset = BOPDataset('./data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)
    image, target, cam_K = dataset[1]
    im_width, im_height = image.size(2), image.size(1)
    image_name = 'test_image'
    if params_config.GRAYSCALE:
        image = image.mean(dim=0, keepdim=True)

    # Set the detection flag
    detection = True

    # Load the models
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
    if detection:
        # Load the detection model
        detection_model = MaskRCNN.load_from_checkpoint(
            './surfemb/patch_extraction/checkpoints/best-checkpoint.ckpt',
            num_classes=dataset.num_classes,
            learning_rate=params_config.LEARNING_RATE
        ).to(params_config.DEVICE)

        # Infer the detection model on the image
        preds = infer_detector(detection_model, image)

        # Visualize the detections
        visualize_detections(image, target, './results', image_name, preds)

    # If not detection, use the target as the preds
    else:
        # Get the synthetic preds from the dataset target --> preds = target
        preds = target.copy()

        # Boxes have to be squared and be divisible by 32
        for i in range(len(preds['boxes'])):
            x_min, y_min, x_max, y_max = preds['boxes'][i]
            # Update the box
            preds['boxes'][i] = torch.tensor(calculate_new_bounding_box(im_width, im_height, x_min, y_min, x_max, y_max))
        
        preds['boxes'] = preds['boxes'].int().to(params_config.DEVICE)
        preds['labels'] = preds['labels'].to(params_config.DEVICE)

        # Visualize the targets
        visualize_detections(image, target, './results', image_name)

    # Infer the surfemb model on the targets
    results = infer_surfemb(image, image_name, cam_K, preds, embedding_model, objs, obj_ids, surface_samples, surface_sample_normals, renderer, res_crop)

    target_string = target_to_string(target)
    results_string = results_to_string(results)
    print_side_by_side(target_string, results_string)

