from ..custom_patch_extraction.dataset import BOPDataset
from ..custom_patch_extraction.model import MaskRCNN
from ..custom_patch_extraction.utils import infer_detector
from ..surface_embedding import SurfaceEmbeddingModel
from ..data.renderer import ObjCoordRenderer
from ..data.obj import load_objs
from .. import utils
from .. import pose_est
from .. import pose_refine
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.utils.data
import json

# Create Dataset and get image
dataset = BOPDataset('./data/bop/itodd', subset='train_pbr', split='test', test_ratio=0.1)
image, target, cam_K = dataset[1]
image = image.mean(dim=0, keepdim=True)
greyscale = True

'''
# Visualize the data
if greyscale:
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
else:
    plt.imshow(image.permute(1, 2, 0))
for box in target['boxes']:
    plt.plot([box[0], box[2], box[2], box[0], box[0]], [box[1], box[1], box[3], box[3], box[1]], 'g')
plt.show()
'''

# Load the models
detection_model = MaskRCNN.load_from_checkpoint(
    './surfemb/custom_patch_extraction/checkpoints/best-checkpoint.ckpt',
    num_classes=dataset.num_classes,
    learning_rate=0.001
)

model = SurfaceEmbeddingModel.load_from_checkpoint('./data/models/itodd-3qnq15p6.compact.ckpt')
model.eval()
model.freeze()

# Get the objects and surface samples
objs, obj_ids = load_objs(Path('./data/bop/itodd/models'))
surface_samples, surface_sample_normals = utils.load_surface_samples('itodd', obj_ids)

# Create the renderer
res_crop = 224
renderer = ObjCoordRenderer(objs, res_crop)

# Infer the detection model on the image
preds = infer_detector(detection_model, image)

'''
# visualize the detections 
if greyscale:
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
else:
    plt.imshow(image.permute(1, 2, 0))
for box in preds['boxes']:
    plt.plot([box[0], box[2], box[2], box[0], box[0]], [box[1], box[1], box[3], box[3], box[1]], 'r')
plt.show()
'''

# initialize opencv windows
cols = 4
window_names = 'img', 'mask_est', 'queries', 'keys', \
               'pose', 'mask_score', 'coord_score', 'query_norm'
for j, name in enumerate(window_names):
    row = j // cols
    col = j % cols
    cv2.imshow(name, np.zeros((res_crop, res_crop)))
    cv2.moveWindow(name, 100 + 300 * col, 100 + 300 * row)

print()
print('With an opencv window active:')
print("press 'a', 'd' and 'x'(random) to get a new input image,")
print("press 'e' to estimate pose, and 'r' to refine pose estimate,")
print("press 'g' to see the ground truth pose,")
print("press 'q' to quit.")

# Iterate over the Image crops
data_i = 0
results = []
while True:
    print()
    print('------------ new input -------------')
    # Take out an image crop
    obj_idx = preds['labels'][data_i].item() - 1 # surface embedding model uses 0-based indexing
    box = preds['boxes'][data_i]
    img = image[:, box[1]:box[3], box[0]:box[2]]
    old_width, old_height = img.shape[1], img.shape[2]
    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(res_crop, res_crop), mode='bilinear', align_corners=False).squeeze(0)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8) # Convert to uint8

    # Get scaling factor
    scale_x = res_crop / old_width
    scale_y = res_crop / old_height
    assert scale_x == scale_y
    scale = scale_x

    # Correct the camera matrix --> apperently not needed
    K_crop = np.copy(cam_K)
    
    obj_ = objs[obj_idx]

    print(f'i: {data_i}, obj_id: {obj_ids[obj_idx]}')

    with utils.timer('forward_cnn'):
        mask_lgts, query_img = model.infer_cnn(img, obj_idx, rotation_ensemble=False)

    mask_prob = torch.sigmoid(mask_lgts)
    query_vis = model.get_emb_vis(query_img)
    query_norm_img = torch.norm(query_img, dim=-1) * mask_prob
    query_norm_img /= query_norm_img.max()
    cv2.imshow('query_norm', query_norm_img.cpu().numpy())

    dist_img = torch.zeros(res_crop, res_crop, device=model.device)

    verts_np = surface_samples[obj_idx]
    verts = torch.from_numpy(verts_np).float()
    normals = surface_sample_normals[obj_idx]
    verts_norm = (verts_np - obj_.offset) / obj_.scale
    with utils.timer('forward_mlp'):
        keys_verts = model.infer_mlp(torch.from_numpy(verts_norm).float().to(model.device), obj_idx)  # (N, emb_dim)
    keys_means = keys_verts.mean(dim=0)  # (emb_dim,)

    # visualize
    img_vis = img[..., ::-1].astype(np.float32) / 255

    cv2.imshow('img', img_vis)
    cv2.imshow('mask_est', torch.sigmoid(mask_lgts).cpu().numpy())
    cv2.imshow('queries', query_vis.cpu().numpy())

    last_mouse_pos = 0, 0
    uv_pts_3d = []
    current_pose = None
    down_sample_scale = 3

    def mouse_cb(event, x, y, flags=0, *_):
        global last_mouse_pos


    for name in window_names:
        cv2.setMouseCallback(name, mouse_cb)


    def debug_pose_hypothesis(R, t, obj_pts=None, img_pts=None):
        global uv_pts_3d, current_pose
        current_pose = R, t
        render = renderer.render(obj_idx, cam_K, R, t)
        render_mask = render[..., 3] == 1.
        pose_img = img_vis.copy()
        if greyscale:
            pose_img[render_mask] = pose_img[render_mask] * 0.5 + render[..., :1][render_mask] * 0.25 + 0.5
        else:
            pose_img[render_mask] = pose_img[render_mask] * 0.5 + render[..., :3][render_mask] * 0.25 + 0.25

        if obj_pts is not None:
            colors = np.eye(3)[::-1]
            for (x, y), c in zip(img_pts.astype(int), colors):
                cv2.drawMarker(pose_img, (x, y), tuple(c), cv2.MARKER_CROSS, 10)
            uv_pts_3d = obj_pts
            mouse_cb(None, *last_mouse_pos)

        cv2.imshow('pose', pose_img)

        poses = np.eye(4)
        poses[:3, :3] = R
        poses[:3, 3:] = t
        pose_est.estimate_pose(
            mask_lgts=mask_lgts, query_img=query_img,
            obj_pts=verts, obj_normals=normals, obj_keys=keys_verts,
            obj_diameter=obj_.diameter, K=cam_K, down_sample_scale=down_sample_scale,
            visualize=True, poses=poses[None],
        )

    def estimate_pose():
        print()
        with utils.timer('pnp ransac'):
            R, t, scores, mask_scores, coord_scores, dist_2d, size_mask, normals_mask = pose_est.estimate_pose(
                mask_lgts=mask_lgts, query_img=query_img, down_sample_scale=down_sample_scale,
                obj_pts=verts, obj_normals=normals, obj_keys=keys_verts,
                obj_diameter=obj_.diameter, K=cam_K,
            )
        if not len(scores):
            print('no pose')
            return None
        else:
            R, t, scores, mask_scores, coord_scores = [a.cpu().numpy() for a in
                                                       (R, t, scores, mask_scores, coord_scores)]
            best_pose_idx = np.argmax(scores)
            R_, t_ = R[best_pose_idx], t[best_pose_idx, :, None]
            debug_pose_hypothesis(R_, t_)
            return R_, t_


    while True:
        print()
        key = cv2.waitKey()
        if key == ord('q'):
            # Save the results
            with open('results.json', 'w') as f:
                json.dump(results, f)
            quit()
        elif key == ord('s'): # Save the results of the current crop
            # TODO: Correct the estimated pose so that it corresponds to the whole image and not just the crop

            # Append the corrected pose to results
            results.append({
                'obj_id': obj_ids[obj_idx],
                'K_crop': K_crop.tolist(),
                'R': current_pose[0].tolist(),
                't': current_pose[1].tolist(),
                'box': box.tolist(),
                'scale': scale
            })
            print('Corrected pose results saved')
        elif key == ord('a'):
            data_i = (data_i - 1) % len(preds['boxes'])
            break
        elif key == ord('d'):
            data_i = (data_i + 1) % len(preds['boxes'])
            break
        elif key == ord('x'):
            data_i = np.random.randint(len(preds['boxes']))
            break
        elif key == ord('e'):
            print('pose est:')
            estimate_pose()
        elif key == ord('r'):
            print('refine:')
            if current_pose is not None:
                with utils.timer('refinement'):
                    R, t, score_r = pose_refine.refine_pose(
                        R=current_pose[0], t=current_pose[1], query_img=query_img, keys_verts=keys_verts,
                        obj_idx=obj_idx, obj_=obj_, K_crop=cam_K, model=model, renderer=renderer,
                    )
                    trace = np.trace(R @ current_pose[0].T)
                    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                    print(f'refinement angle diff: {np.rad2deg(angle):.1f} deg')
                debug_pose_hypothesis(R, t)

        mouse_cb(None, *last_mouse_pos)