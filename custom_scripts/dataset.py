""" Contains the Datasets TLESS and ITODD. """

import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
from tqdm import tqdm

class TLESS(Dataset):
    """ Dataset class for TLESS. """

    def __init__(self, root_dir, split='train_pbr'):
        """ Args:
            root_dir (string): Directory with all the images.
            split (string): 'train_pbr' or 'test_primesense'.
        """
        self.root_dir = root_dir
        self.split = split
        subdirs = os.listdir(os.path.join(root_dir, split))
        self.subdirs = sorted(subdirs, key=lambda x: int(x))
        self.rgb_paths = []
        self.mask_paths = []
        self.scene_gts = []
        self.scene_gt_infos = []

        # Iterate over all subdirs
        for subdir in tqdm(self.subdirs, desc='Loading dataset'):
            rgb_path = os.path.join(root_dir, split, subdir, 'rgb')

            # Open gt Files belonging to the current subdir
            scene_gt = json.load(open(os.path.join(root_dir, split, subdir, 'scene_gt.json')))
            scene_gt_info = json.load(open(os.path.join(root_dir, split, subdir, 'scene_gt_info.json')))

            # Append the paths and scene_gt to the lists
            for file in sorted(os.listdir(rgb_path), key=lambda x: int(x.split('.')[0])):
                self.rgb_paths.append(os.path.join(rgb_path, file))
                filename = file.split('.')[0]
                image_id = int(filename)
                self.scene_gts.append(scene_gt[f'{image_id}'])
                self.scene_gt_infos.append(scene_gt_info[f'{image_id}'])
                
    def __len__(self):
        return len(self.rgb_paths)
    
    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        rgb = read_image(rgb_path)
        scene_gt = self.scene_gts[idx]
        scene_gt_infos = self.scene_gt_infos[idx]

        # Extract the masks
        subdir = rgb_path.split('/')[-1].split('.')[0]
        mask_path = os.path.join(self.root_dir, self.split, subdir, 'mask')
        filename = rgb_path.split('/')[-1].split('.')[0]
        mask = torch.zeros(20, 540, 720)
        for file in os.listdir(mask_path):
            if file.startswith(filename):
                mask_id = int(file.split('.')[0].split('_')[-1])
                mask[mask_id] = read_image(os.path.join(mask_path, file))

        return rgb, mask, scene_gt, scene_gt_infos

if __name__ == '__main__':
    tless = TLESS('../data/bop/tless', split='train_pbr')
    rgb, mask, scene_gt, scene_gt_infos = tless[0]
    
    # Print all the different values in mask
    print(mask[0].shape)
    print(mask[3].unique())
