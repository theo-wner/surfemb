from ..patch_extraction.dataset import BOPDataset
from ..patch_extraction.model import MaskRCNN
from ..patch_extraction.utils import infer_detector, visualize_detections
from ..surface_embedding import SurfaceEmbeddingModel
from ..data.renderer import ObjCoordRenderer
from ..data.obj import load_objs
from .. import utils
from . import params_config
from .custom_infer import infer_surfemb
from .debug_estimated_poses import target_to_string, results_to_string, ensemble_results_to_string, print_side_by_side
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
    num_objs = len(predictions[0]['objects'])
    all_ts = np.zeros((len(predictions), num_objs, 3))
    all_eulers = np.zeros((len(predictions), num_objs, 3))
    obj_ids = [object['obj_id'] for object in predictions[0]['objects']]
    for i in range(num_objs):
        for j in range(len(predictions)):
            all_ts[j, i] = np.array(predictions[j]['objects'][i]['t']).flatten()
            all_eulers[j, i] = np.array(predictions[j]['objects'][i]['euler']).flatten()

    ensemble_results = []
    for i in range(num_objs):
        mean_t = np.mean(all_ts[:, i], axis=0)
        std_t = np.std(all_ts[:, i], axis=0)
        mean_euler = np.mean(all_eulers[:, i], axis=0)
        std_euler = np.std(all_eulers[:, i], axis=0)
        ensemble_results.append({
            'obj_id': obj_ids[i],
            't_mean': mean_t.tolist(),
            't_std': std_t.tolist(),
            'euler_mean': mean_euler.tolist(),
            'euler_std': std_euler.tolist()
        })

    return ensemble_results

if __name__ == '__main__':
    # Create Dataset and get image
    dataset = BOPDataset('./data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)
    image, target, cam_K = dataset[2]
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

    # Visualize the detections
    visualize_detections(image, target, './results', image_name, preds)

    # Load the ensemble models
    ensemble_models = load_models('./data/models/ensemble')

    # Infer the surfemb model on the predictions
    results = predict_with_ensemble(ensemble_models, image, image_name, cam_K, preds, objs, obj_ids, surface_samples, surface_sample_normals, renderer, res_crop)

    # Print the results
    target_string = target_to_string(target)
    ensemble_results_string = ensemble_results_to_string(results)
    print_side_by_side(target_string, ensemble_results_string)



