import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from transformers import SamModel, SamConfig, SamProcessor
from skimage import io, transform
import torch.nn.functional as F
import argparse
import tifffile
import cv2


# Function that inputs the output and plots image and mask
def show_output(result_dict, axes=None):
     if axes:
        ax = axes
     else:
        ax = plt.gca()
        ax.set_autoscale_on(False)
     sorted_result = sorted(result_dict, key=(lambda x: x['area']),      reverse=True)
     # Plot for each segment area
     for val in sorted_result:
        mask = val['segmentation']
        img = np.ones((mask.shape[0], mask.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, mask*0.5)))

# Give the path of your image
IMAGE_PATH= 'assets/img_demo.png'
# Read the image from the path
image= cv2.imread(IMAGE_PATH)
# Convert the image from BGR (OpenCV format) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Convert the numpy array to a tensor
image_tensor = torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)

sam_checkpoint = "work_dir/SAM/sam_vit_b_01ec64.pth"
#sam_h_checkpoint = "sam_vit_h_4b8939.pth"
med_sam_checkpoint = "work_dir/SAM/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"
#sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# Load the model configuration
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
# Create an instance of the model architecture with the loaded configuration
med_sam_model = SamModel(config=model_config)
#state_dict = torch.load(med_sam_checkpoint, map_location=torch.device('cpu'))
#med_sam_model.load_state_dict(state_dict, strict=False)
med_sam_model.to(device=device)

"""
input_points (torch.FloatTensor of shape (batch_size, num_points, 2)) —
Input 2D spatial points, this is used by the prompt encoder to encode the prompt.
Generally yields to much better results. The points can be obtained by passing a
list of list of list to the processor that will create corresponding torch tensors
of dimension 4. The first dimension is the image batch size, the second dimension
is the point batch size (i.e. how many segmentation masks do we want the model to
predict per input point), the third dimension is the number of points per segmentation
mask (it is possible to pass multiple points for a single mask), and the last dimension
is the x (vertical) and y (horizontal) coordinates of the point. If a different number
of points is passed either for each image, or for each mask, the processor will create
“PAD” points that will correspond to the (0, 0) coordinate, and the computation of the
embedding will be skipped for these points using the labels.

"""
# Define the size of your array
array_size = 512

# Define the size of your grid
grid_size = 10

# Generate the grid points
x = np.linspace(0, array_size-1, grid_size)
y = np.linspace(0, array_size-1, grid_size)

# Generate a grid of coordinates
xv, yv = np.meshgrid(x, y)

# Convert the numpy arrays to lists
xv_list = xv.tolist()
yv_list = yv.tolist()

# Combine the x and y coordinates into a list of list of lists
input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]

#We need to reshape our nxn grid to the expected shape of the input_points tensor
# (batch_size, point_batch_size, num_points_per_image, 2),
# where the last dimension of 2 represents the x and y coordinates of each point.
#batch_size: The number of images you're processing at once.
#point_batch_size: The number of point sets you have for each image.
#num_points_per_image: The number of points in each set.
input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)

inputs = processor(image_tensor, input_points=input_points, return_tensors="pt")
# Move the input tensor to the GPU if it's not already there
inputs = {k: v.to(device) for k, v in inputs.items()}
med_sam_model.eval()

# forward pass
with torch.no_grad():
  outputs = med_sam_model(**inputs, multimask_output=False)

# apply sigmoid
prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
prob = prob.cpu().numpy().squeeze()
prediction = (prob > 0.5).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title("Image")
axes[1].imshow(prob)
axes[1].set_title("Probability Map")
axes[2].imshow(prediction, cmap='gray')
axes[2].set_title("Prediction")
# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
plt.savefig("segment.png")