import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import os
import shutil
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
import matplotlib.patches as patches
import torch

def get_model(num_classes):
    # Load a pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(weights=True)
    # Update the classifier to match the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    # Update the mask predictor to match the number of classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model

def visualize_data(rgb, targets, preds=None):
    # Delete content of the figures folder
    folder = './custom_patch_extraction/figures/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    boxes_t = targets['boxes']
    labels_t = targets['labels']
    masks_t = targets['masks']
    masks_t = (masks_t * 255).byte()

    # Save RGB image and masks as single layer binary images for visualization
    rgb = ToPILImage()(rgb)
    rgb.save('./custom_patch_extraction/figures/rgb.png')
    for i in range(masks_t.size(0)):
        mask_img_t = Image.fromarray((masks_t[i].numpy()).astype(np.uint8))
        mask_img_t.save(f'./custom_patch_extraction/figures/target_mask_{i}_object_{labels_t[i]}.png')
    
    # Visualize the bounding boxes
    fig, ax = plt.subplots(1)
    ax.imshow(rgb)
    for i in range(boxes_t.size(0)):
        box = boxes_t[i]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1], f'Object {labels_t[i]}', color='g')
    plt.axis('off')
    plt.savefig('./custom_patch_extraction/figures/target_boxes.png')
    plt.close() 

    # Do the same for the predictions if available
    if preds is not None:
        boxes_p = preds['boxes']
        labels_p = preds['labels']
        masks_p = preds['masks']
        masks_p = (masks_p * 255).byte()
        for i in range(masks_p.size(0)):
            mask_img_p = Image.fromarray((masks_p[i].numpy()).astype(np.uint8))
            mask_img_p.save(f'./custom_patch_extraction/figures/pred_mask_{i}_object_{labels_p[i]}.png')
        fig, ax = plt.subplots(1)
        ax.imshow(rgb)
        for i in range(boxes_p.size(0)):
            box = boxes_p[i]
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1], f'Object {labels_p[i]}', color='r')
        plt.axis('off')
        plt.savefig('./custom_patch_extraction/figures/pred_boxes.png')
        plt.close()

        # Save target and pred boxes togheter in a new image  next to each other
        target_boxes = Image.open('./custom_patch_extraction/figures/target_boxes.png')

        pred_boxes = Image.open('./custom_patch_extraction/figures/pred_boxes.png')
        width, height = target_boxes.size

        new_im = Image.new('RGB', (2 * width, height))
        new_im.paste(target_boxes, (0, 0))
        new_im.paste(pred_boxes, (width, 0))
        new_im.save('./custom_patch_extraction/figures/target_pred_boxes.png')

def compute_iou(box1, box2):
    """
    Compute IoU (Intersection over Union) for two boxes.
    Each box is represented as [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0

    return inter_area / union_area

def filter_boxes(preds, iou_threshold):
    keep = torch.ones(len(preds['boxes']), dtype=torch.bool)
    for i in range(len(preds['boxes'])):
        if keep[i]:
            for j in range(i + 1, len(preds['boxes'])):
                if keep[j]:
                    iou = compute_iou(preds['boxes'][i].cpu().numpy(), preds['boxes'][j].cpu().numpy())
                    if iou > iou_threshold:
                        if preds['scores'][i] > preds['scores'][j]:
                            keep[j] = 0
                        else:
                            keep[i] = 0
                            break
    preds['boxes'] = preds['boxes'][keep].int()

    # Ensure that all the boxes are quadratic --> make them larger if necessary
    im_width, im_height = preds['masks'].size(3), preds['masks'].size(2)
    for i in range(len(preds['boxes'])):
        box = preds['boxes'][i]
        width = box[2] - box[0]
        height = box[3] - box[1]
        if width > height:
            diff = width - height
            box[1] = max(0, box[1] - diff // 2)
            box[3] = min(im_height, box[3] + diff - diff // 2)
        else:
            diff = height - width
            box[0] = max(0, box[0] - diff // 2)
            box[2] = min(im_width, box[2] + diff - diff // 2)
        preds['boxes'][i] = box
        
    # Pad the boxes to be divisible by 32 and within the image
    preds['boxes'][:, 0] = torch.clamp(preds['boxes'][:, 0] - (preds['boxes'][:, 0] % 32), min=0, max=im_width)
    preds['boxes'][:, 1] = torch.clamp(preds['boxes'][:, 1] - (preds['boxes'][:, 1] % 32), min=0, max=im_height)
    preds['boxes'][:, 2] = torch.clamp(preds['boxes'][:, 2] + (32 - (preds['boxes'][:, 2] % 32)), min=0, max=im_width)
    preds['boxes'][:, 3] = torch.clamp(preds['boxes'][:, 3] + (32 - (preds['boxes'][:, 3] % 32)), min=0, max=im_height)

    preds['labels'] = preds['labels'][keep]
    preds['masks'] = preds['masks'][keep].squeeze(dim=1)
    preds['scores'] = preds['scores'][keep]
    return preds

def infer_detector(model, image):
    """
    Infer the model on the given image.
    Args:
        model: The model to infer.
        image: The image to infer on.
    Both the model and the image should be on the same device.
    """ 
    model.eval() # Set the model to evaluation mode
    image = image.unsqueeze(0) # Add a batch dimension
    with torch.no_grad(): # Disable gradient computation
        preds = model(image)[0] # Make predictions, 0 because batch size is 1
    iou_threshold = 0.2 # Set the IoU threshold for filtering overlapping boxes
    preds = filter_boxes(preds, iou_threshold) # Filter overlapping boxes
    return preds