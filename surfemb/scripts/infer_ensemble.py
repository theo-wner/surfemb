from ..patch_extraction.dataset import BOPDataset
from ..patch_extraction.model import MaskRCNN
from ..patch_extraction.utils import infer_detector
from ..surface_embedding import SurfaceEmbeddingModel
from ..data.renderer import ObjCoordRenderer
from ..data.obj import load_objs
from .. import utils
from . import params_config
from .custom_infer import infer_surfemb
import torch
from pathlib import Path
import os

def load_models(checkpoint_path):
    models = []
    for path in os.listdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, path)
        model = SurfaceEmbeddingModel.load_from_checkpoint(model_path).eval().to(params_config.DEVICE).freeze()
        models.append(model)
    return models

def predict_with_ensemble(models, image, image_name, cam_K, preds, objs, obj_ids, surface_samples, surface_sample_normals, renderer, res_crop):
    predictions = []
    for model in models:
        with torch.no_grad(): 
            results = infer_surfemb(image, image_name, cam_K, preds, model, objs, obj_ids, surface_samples, surface_sample_normals, renderer, res_crop)
            predictions.append(results)
    return predictions

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
    ensemble_models = load_models('./data/models')

    # Infer the surfemb model on the predictions
    results = predict_with_ensemble(ensemble_models, image, image_name, cam_K, preds, objs, obj_ids, surface_samples, surface_sample_normals, renderer, res_crop)


