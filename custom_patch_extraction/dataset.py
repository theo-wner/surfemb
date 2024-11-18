""" Contains the Datasets TLESS and ITODD. """

import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from utils import visualize_data

class BOPDataset(Dataset):
    """ Dataset class for TLESS. """

    def __init__(self, root_dir, split='train_pbr'):
        """ Args:
            dataset (string): 'tless' or 'itodd'.
            root_dir (string): Directory with all the images.
            split (string): 'train_pbr' or 'test_primesense' for TLESS
                            'train_pbr' or 'test' for ITODD.
        """
        self.dataset = root_dir.split('/')[-1]
        if self.dataset == 'tless':
            self.height = 540
            self.width = 720
            self.num_classes = 31
        elif self.dataset == 'itodd':
            self.height = 960
            self.width = 1280
            self.num_classes = 28

        self.root_dir = root_dir
        self.split = split
        subdirs = os.listdir(os.path.join(root_dir, split))
        self.subdirs = sorted(subdirs, key=lambda x: int(x))
        self.rgb_paths = []

        # Iterate over all subdirs
        for subdir in self.subdirs:
            rgb_path = os.path.join(root_dir, split, subdir, 'rgb')

            # Append the paths and scene_gt to the lists
            for file in sorted(os.listdir(rgb_path), key=lambda x: int(x.split('.')[0])):
                self.rgb_paths.append(os.path.join(rgb_path, file))
                
    def __len__(self):
        return len(self.rgb_paths)
    
    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        rgb = read_image(rgb_path)
        rgb = rgb.float() / 255.0

        subdir = rgb_path.split('/')[-3]
        image_id = int(rgb_path.split('/')[-1].split('.')[0])

        scene_gt = json.load(open(os.path.join(self.root_dir, self.split, subdir, 'scene_gt.json')))[str(image_id)]
        scene_gt_info = json.load(open(os.path.join(self.root_dir, self.split, subdir, 'scene_gt_info.json')))[str(image_id)]

        mask_path = os.path.join(self.root_dir, self.split, subdir, 'mask_visib')
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

        return rgb, targets

    def get_boxes_labels_masks(self, scene_gt, scene_gt_info, mask_stack):
        """ Returns the bounding boxes, labels, and masks for the visible objects. """
        boxes = []
        labels = []
        masks = []

        for obj_cnt in range(len(scene_gt)):
            object = scene_gt[obj_cnt]
            object_info = scene_gt_info[obj_cnt]
            box = self.convert_box(object_info['bbox_visib'])
            if object_info['px_count_visib'] > 0 and self.box_is_valid(box):
                boxes.append(box)
                labels.append(object['obj_id'])
                masks.append(mask_stack[obj_cnt])  # Ensure alignment with mask_stack

        if len(boxes) > 0:
            masks = torch.stack(masks, dim=0)  # Stack masks into a tensor [N, H, W]
        else:
            masks = torch.empty((0, self.height, self.width), dtype=torch.float32)

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
            masks
        )

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
    tless = BOPDataset('../data/bop/tless', split='test_primesense')

    rgb, targets = tless[0]
    
    boxes = targets['boxes']
    labels = targets['labels']
    mask = targets['masks']

    visualize_data(rgb, targets)

        
