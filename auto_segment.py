import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from skimage import io, transform
import torch.nn.functional as F
import argparse
import tifffile
import cv2


# Function that inputs the output and plots image and mask
def show_output(anns, ax=None):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    ax.imshow(img)

# Give the path of your image
IMAGE_PATH= 'assets/img_demo.png'
# Read the image from the path
image= cv2.imread(IMAGE_PATH)
# Convert the image from BGR (OpenCV format) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "sam_vit_b_01ec64.pth"
#sam_checkpoint = "work_dir/MedSAM/medsam_vit_b.pth"
#sam_h_checkpoint = "sam_vit_h_4b8939.pth"
med_sam_checkpoint = "MedSAM/medsam_vit_b.pth"
model_type = "vit_b"
device = "cuda"
med_sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
state_dict = torch.load(med_sam_checkpoint, map_location=torch.device('cpu'))
med_sam_model.load_state_dict(state_dict)
med_sam_model.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=med_sam_model,
    points_per_side=32,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.5,
    crop_n_layers=0,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
masks = mask_generator.generate(image_rgb)

fig, axes = plt.subplots(1, 2, figsize=(20,10))
axes[0].imshow(image)
axes[0].axis('off')
axes[1].imshow(np.zeros_like(image))
axes[1].axis('off')
show_output(masks, axes[1])
plt.axis('off')
plt.savefig("auto_segment.png")