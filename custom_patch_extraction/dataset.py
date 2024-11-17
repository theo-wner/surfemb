""" Contains the Datasets TLESS and ITODD. """

import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

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
            self.num_classes = 20
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

        subdir = rgb_path.split('/')[-3]
        image_id = int(rgb_path.split('/')[-1].split('.')[0])

        scene_gt = json.load(open(os.path.join(self.root_dir, self.split, subdir, 'scene_gt.json')))[str(image_id)]
        scene_gt_info = json.load(open(os.path.join(self.root_dir, self.split, subdir, 'scene_gt_info.json')))[str(image_id)]
        boxes, labels = self.get_boxes_labels(scene_gt, scene_gt_info)

        mask_path = os.path.join(self.root_dir, self.split, subdir, 'mask_visib')
        mask_stack = torch.zeros(self.num_classes, self.height, self.width)
        for file in os.listdir(mask_path):
            if file.startswith(subdir):
                mask_id = int(file.split('.')[0].split('_')[-1])
                mask_stack[mask_id] = read_image(os.path.join(mask_path, file))
        mask = self.mask_stack_2_mask(mask_stack, scene_gt)

        return rgb, {'boxes': boxes, 'labels': labels, 'masks': mask}
    
    def mask_stack_2_mask(self, mask_stack, scene_gt):
        """ Converts the 20-channel mask stack to a single channel mask 
            according to the object ids in scene_gt.
        """
        mask = torch.zeros(1, self.height, self.width, dtype=torch.uint8)
        obj_cnt = 0
        for object in scene_gt:
            obj_id = object['obj_id']
            mask_layer = mask_stack[obj_cnt].unsqueeze(0)
            mask[mask_layer==255] = obj_id
            obj_cnt += 1
        return mask
    
    def get_boxes_labels(self, scene_gt, scene_gt_info):
        """ Returns the bounding boxes and labels for the visible objects. """
        boxes = []
        labels = []
        for obj_cnt in range(len(scene_gt)):
            object = scene_gt[obj_cnt]
            object_info = scene_gt_info[obj_cnt]
            if object_info['px_count_visib'] != 0:
                boxes.append(object_info['bbox_visib'])
                labels.append(object['obj_id'])
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

if __name__ == '__main__':
    tless = BOPDataset('../data/bop/tless', split='train_pbr')

    rgb, targets = tless[0]
    print(rgb.dtype)
    print(targets['boxes'].dtype)
    print(targets['labels'].dtype)
    print(targets['masks'].dtype)


        
