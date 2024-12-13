from pathlib import Path

import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from . import params_config
from .. import utils
from ..data import obj, instance
from ..data.config import config
from ..surface_embedding import SurfaceEmbeddingModel

def worker_init_fn(*_, seed):
    # each worker should only use one os thread
    # numpy/cv2 takes advantage of multithreading by default
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import cv2
    cv2.setNumThreads(0)

    # random seed
    import numpy as np
    np.random.seed(seed)

def train_member(seed):
    # load objs
    root = Path('./data/bop/itodd')
    cfg = config['itodd']
    objs, obj_ids = obj.load_objs(root / 'models')
    assert len(obj_ids) > 0

    # model
    model = SurfaceEmbeddingModel(n_objs=len(obj_ids), seed=seed)

    # datasets
    auxs = model.get_auxs(objs, params_config.RES_CROP)

    data = instance.BopInstanceDataset(
            dataset_root=root, pbr=True, test=False, cfg=cfg, obj_ids=obj_ids, auxs=auxs,
            min_visib_fract=0.1, scene_ids=None,
        )

    n_valid = 200
    data_train, data_valid = torch.utils.data.random_split(
        data, (len(data) - n_valid, n_valid),
        generator=torch.Generator().manual_seed(seed),
    )

    loader_args = dict(
        batch_size=params_config.BATCH_SIZE,
        num_workers=params_config.NUM_WORKERS,
        persistent_workers=True, shuffle=True,
        worker_init_fn=worker_init_fn(seed=seed), pin_memory=True,
    )
    loader_train = torch.utils.data.DataLoader(data_train, drop_last=True, **loader_args)
    loader_valid = torch.utils.data.DataLoader(data_valid, **loader_args)

    # train
    log_dir = Path('data/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir = 'data/logs'   
    run = wandb.init(
        project='surfemb',
        dir=log_dir,
        name=f'{wandb.util.generate_id()}-seed-{seed}',  # Unique name for each run
        reinit=True  # Important for independent runs
    )

    logger = pl.loggers.WandbLogger(experiment=run)

    model_ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath='data/models/', save_top_k=0, save_last=True)
    model_ckpt_cb.CHECKPOINT_NAME_LAST = f'itodd-{run.id}-seed-{seed}'
    trainer = pl.Trainer(
        logger=logger, gpus=[params_config.DEVICE_NUM], max_steps=params_config.MAX_STEPS,
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            model_ckpt_cb,
        ],
        val_check_interval=min(1., n_valid / len(data) * 50)  # spend ~1/50th of the time on validation
    )
    '''
    # Print some weights to see if the seed is working
    state_dict = model.state_dict()
    cnn_weights = state_dict['cnn.base_model.layer2.0.conv2.weight']
    mlp_weights = state_dict['mlps.8.net.3.weight']
    print(f'CNN Weights Seed {seed}: {cnn_weights[0, :10, 0, 0]}')
    print(f'MLP Weights Seed {seed}: {mlp_weights[0, :10]}')
    '''
    trainer.fit(model, loader_train, loader_valid)

if __name__ == '__main__':
    # Train 10 Ensemble members
    for seed in range(1, 10):
        train_member(seed)