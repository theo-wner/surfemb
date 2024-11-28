""" Contains the Datasets TLESS and ITODD. """

import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from custom_patch_extraction.utils import visualize_data

class BOPDataset(Dataset):
    """ Dataset class for TLESS. """

    def __init__(self, root_dir, subset='train_pbr', split='train', test_ratio=0.3):
        """ Args:
            root_dir (string): Directory with all the images.
            subset (string): 'train_pbr' or 'test_primesense' for TLESS
                            'train_pbr' or 'test' for ITODD.
            split (string): 'train' or 'test'.
            test_ratio (float): Ratio of the dataset to use for testing.
        """
        self.dataset = root_dir.split('/')[-1]
        if self.dataset == 'tless':
            self.height = 540
            self.width = 720
            self.num_classes = 31
        elif self.dataset == 'itodd':
            self.height = 960
            self.width = 1280
            self.num_classes = 29

        self.root_dir = root_dir
        self.subset = subset
        self.split = split
        self.test_ratio = test_ratio
        subdirs = os.listdir(os.path.join(self.root_dir, self.subset))
        self.subdirs = sorted(subdirs, key=lambda x: int(x))
        self.rgb_paths = []

        # Iterate over all subdirs
        for subdir in self.subdirs:
            rgb_path = os.path.join(self.root_dir, self.subset, subdir, 'rgb')

            # Append the paths and scene_gt to the lists
            for file in sorted(os.listdir(rgb_path), key=lambda x: int(x.split('.')[0])):
                self.rgb_paths.append(os.path.join(rgb_path, file))

        # take out a subset of the dataset for testing (30%) 
        test_size = int(len(self.rgb_paths) * test_ratio)
        random.seed(42) # 42 is the answer to everything
        random_indices = sorted(random.sample(range(len(self.rgb_paths)), test_size))
        self.rgb_paths_test = [self.rgb_paths[i] for i in random_indices]
        self.rgb_paths_train = [file for i, file in enumerate(self.rgb_paths) if i not in random_indices]


    def __len__(self):
        if self.split == 'train':
            return len(self.rgb_paths_train)
        elif self.split == 'test':
            return len(self.rgb_paths_test)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            rgb_path = self.rgb_paths_train[idx]
        elif self.split == 'test':
            rgb_path = self.rgb_paths_test[idx]

        rgb = read_image(rgb_path)
        rgb = rgb.float() / 255.0

        subdir = rgb_path.split('/')[-3]
        image_id = int(rgb_path.split('/')[-1].split('.')[0])

        scene_gt = json.load(open(os.path.join(self.root_dir, self.subset, subdir, 'scene_gt.json')))[str(image_id)]
        scene_gt_info = json.load(open(os.path.join(self.root_dir, self.subset, subdir, 'scene_gt_info.json')))[str(image_id)]
        scene_camera = json.load(open(os.path.join(self.root_dir, self.subset, subdir, 'scene_camera.json')))[str(image_id)]
        cam_K = np.array(scene_camera['cam_K']).reshape(3, 3)

        mask_path = os.path.join(self.root_dir, self.subset, subdir, 'mask_visib')
        relevant_masks = []
        for file in os.listdir(mask_path):
            if int(file.split('_')[0]) == image_id:
                relevant_masks.append(file)
        mask_stack = []
        for file in sorted(relevant_masks, key=lambda x: int(x.split('_')[1].split('.')[0])):
            mask_layer = read_image(os.path.join(mask_path, file)).squeeze(0)
            mask_layer = (mask_layer > 0).float()
            mask_stack.append(mask_layer)

        boxes, labels, masks = self.get_boxes_labels_masks(scene_gt, scene_gt_info, mask_stack)

        targets = {'boxes': boxes, 'labels': labels, 'masks': masks}

        return rgb, targets, cam_K

    def get_boxes_labels_masks(self, scene_gt, scene_gt_info, mask_stack):
        """ Returns the bounding boxes, labels, and masks for the visible objects. """
        boxes = torch.empty((0, 4), dtype=torch.float32)
        labels = torch.empty((0,), dtype=torch.int64)
        masks = torch.empty((0, self.height, self.width), dtype=torch.float32)

        for obj_cnt in range(len(scene_gt)):
            object = scene_gt[obj_cnt]
            object_info = scene_gt_info[obj_cnt]
            box = self.convert_box(object_info['bbox_visib'])
            if object_info['px_count_visib'] > 0 and self.box_is_valid(box):
                boxes = torch.cat((boxes, torch.tensor([box], dtype=torch.float32)))
                labels = torch.cat((labels, torch.tensor([object['obj_id']], dtype=torch.int64)))
                masks = torch.cat((masks, mask_stack[obj_cnt].unsqueeze(0)))
        return boxes, labels, masks

    def convert_box(self, box):
        """ Converts the bounding box from [x, y, w, h] to [x1, y1, x2, y2]. """
        x, y, w, h = box
        return [x, y, x+w, y+h]
    
    def box_is_valid(self, box):
        x_min, y_min, x_max, y_max = box
        if x_max > x_min and y_max > y_min and x_min >= 0 and y_min >= 0 and x_max <= self.width and y_max <= self.height:
            return True
        return False
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))

if __name__ == '__main__':
    itodd_train = BOPDataset('../data/bop/itodd', subset='train_pbr', split='train', test_ratio=0.1)
    itodd_test = BOPDataset('../data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)  

    print(f'Number of training samples: {len(itodd_train)}')
    print(f'Number of test samples: {len(itodd_test)}')

    rgb, targets = itodd_train[1]
    visualize_data(rgb, targets)

    

        
