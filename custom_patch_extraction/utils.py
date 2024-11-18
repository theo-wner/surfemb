import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import os
import shutil
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
import matplotlib.patches as patches

def get_model(num_classes):
    # Load a pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
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

def visualize_data(rgb, targets):
    # Delete content of the figures folder
    folder = './figures/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    boxes = targets['boxes']
    labels = targets['labels']
    masks = targets['masks']
    # Rescale masks to [0 255] and convert to uint8
    masks = (masks * 255).byte()

    # Save RGB image and masks as single layer binary images for visualization
    rgb = ToPILImage()(rgb)
    rgb.save('./figures/rgb.png')
    for i in range(masks.size(0)):
        mask_img = Image.fromarray((masks[i].numpy()).astype(np.uint8))
        mask_img.save(f'./figures/mask_{i}_object_{labels[i]}.png')
    
    # Visualize the bounding boxes
    fig, ax = plt.subplots(1)
    ax.imshow(rgb)
    for i in range(boxes.size(0)):
        box = boxes[i]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1], f'Object {labels[i]}', color='r')
    plt.axis('off')
    plt.savefig('./figures/boxes.png')
    plt.show()

    # Print the labels
    print(f"Labels: {labels}")