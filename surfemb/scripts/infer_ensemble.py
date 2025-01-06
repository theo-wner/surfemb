from ..patch_extraction.dataset import BOPDataset
from ..patch_extraction.model import MaskRCNN
from ..patch_extraction.utils import infer_detector
from ..surface_embedding import SurfaceEmbeddingModel
from ..data.renderer import ObjCoordRenderer
from ..data.obj import load_objs
from .. import utils
from . import params_config
from .custom_infer import infer_surfemb
from pathlib import Path
import os
import numpy as np

def load_models(checkpoint_path):
    models = []
    for path in os.listdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, path)
        model = SurfaceEmbeddingModel.load_from_checkpoint(model_path).eval().to(params_config.DEVICE)
        model.freeze()
        models.append(model)
    return models

def predict_with_ensemble(models, image, image_name, cam_K, preds, objs, obj_ids, surface_samples, surface_sample_normals, renderer, res_crop):
    predictions = []
    for i in range(len(models)):
        print(f'Infering model {i+1}')
        model = models[i]
        results = infer_surfemb(image, image_name, cam_K, preds, model, objs, obj_ids, surface_samples, surface_sample_normals, renderer, res_crop)
        predictions.append(results)

    # Calculate the mean and std of the predictions
    results_per_object = {}
    for results in predictions:
        for object in results['objects']:
            obj_id = object['obj_id']
            if obj_id not in results_per_object:
                results_per_object[obj_id] = {
                    't_mean': [],
                    'euler_mean': [],
                }
            results_per_object[obj_id]['t_mean'].append(object['t'])
            results_per_object[obj_id]['euler_mean'].append(object['euler'])

    ensemble_results = []
    for obj_id, results in results_per_object.items():
        mean_t = np.mean(results['t_mean'], axis=0).flatten()
        std_t = np.std(results['t_mean'], axis=0).flatten()
        mean_euler = np.mean(results['euler_mean'], axis=0)
        std_euler = np.std(results['euler_mean'], axis=0)
        ensemble_results.append({
            'obj_id': obj_id,
            't_mean': mean_t.tolist(),
            't_std': std_t.tolist(),
            'euler_mean': mean_euler.tolist(),
            'euler_std': std_euler.tolist()
        })

    return ensemble_results

if __name__ == '__main__':
    # Create Dataset and get image
    dataset = BOPDataset('./data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)
    image, target, cam_K = dataset[1]
    image_name = 'test_image'
    if params_config.GRAYSCALE:
        image = image.mean(dim=0, keepdim=True)

    # Load the detection model
    detection_model = MaskRCNN.load_from_checkpoint(
        './surfemb/patch_extraction/checkpoints/best-checkpoint.ckpt',
        num_classes=dataset.num_classes,
        learning_rate=params_config.LEARNING_RATE
    ).to(params_config.DEVICE)

    # Get the objects and surface samples
    objs, obj_ids = load_objs(Path('./data/bop/itodd/models'))
    surface_samples, surface_sample_normals = utils.load_surface_samples('itodd', obj_ids)

    # Create the renderer
    res_crop = 224
    renderer = ObjCoordRenderer(objs, res_crop)

    # Infer the detection model on the image
    preds = infer_detector(detection_model, image)

    # Load the ensemble models
    ensemble_models = load_models('./data/models/ensemble')

    # Infer the surfemb model on the predictions
    results = predict_with_ensemble(ensemble_models, image, image_name, cam_K, preds, objs, obj_ids, surface_samples, surface_sample_normals, renderer, res_crop)

    # Print the results
    for result in results:
        print(f'Object {result["obj_id"]}')
        print(f'Translation:')
        print(f'[{result["t_mean"][0]:.6f}] ± [{result["t_std"][0]:.6f}]')
        print(f'[{result["t_mean"][1]:.6f}] ± [{result["t_std"][1]:.6f}]')
        print(f'[{result["t_mean"][2]:.6f}] ± [{result["t_std"][2]:.6f}]')
        print(f'Euler angles:')
        print(f'[{result["euler_mean"][0]:.6f}] ± [{result["euler_std"][0]:.6f}]')
        print(f'[{result["euler_mean"][1]:.6f}] ± [{result["euler_std"][1]:.6f}]')
        print(f'[{result["euler_mean"][2]:.6f}] ± [{result["euler_std"][2]:.6f}]')
        print()



