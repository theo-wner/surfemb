from ..custom_patch_extraction.dataset import BOPDataset
from ..data.obj import load_objs
from .. import utils
from pathlib import Path
import json
import numpy as np
import cv2

# Load the corresponding image
dataset = BOPDataset('./data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)
image, target, cam_K = dataset[1]

# Save the image in RGB format
rgb = image.permute(1, 2, 0).numpy() * 255
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
cv2.imwrite('image.png', rgb)

# Read the results
with open('results.json', 'r') as f:
    results = json.load(f)

# Load the objects
objs, obj_ids = load_objs(Path('./data/bop/itodd/models'))

# Process the results
for result in results:
    # Create empty image to project the object into
    mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
    obj_id = result['obj_id']
    K_crop = np.array(result['K_crop'])
    box = result['box']
    scale = result['scale']

    R = np.array(result['R'])
    t = np.array(result['t'])

    # Process the pose so that it refers to the whole image and not the crop
    t /= scale
    t[0] += 100
    t[1] += 100

    # Create Projection Matrix P
    pose = np.concatenate((R, t.reshape(3,1)), axis=1)
    P = K_crop @ pose
    
    for obj in objs:
        if obj.obj_id == obj_id:
            for vertex in obj.mesh.vertices:
                vertex = np.append(vertex, 1)
                vertex = np.matmul(P, vertex)
                vertex = vertex / vertex[2]
                vertex = vertex[:2]
                vertex = vertex.astype(int)
                if vertex[0] >= 0 and vertex[0] < mask.shape[1] and vertex[1] >= 0 and vertex[1] < mask.shape[0]:
                    mask[vertex[1], vertex[0]] = 255


    # Save the mask
    cv2.imwrite(f'mask_{obj_id}.png', mask)
    print(f'Mask saved for object {obj_id}')



