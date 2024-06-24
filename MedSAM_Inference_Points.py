# -*- coding: utf-8 -*-
# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from skimage import io, transform
import torch.nn.functional as F
import argparse


# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, cmap='gray')


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W, input_points):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=input_points,
        boxes=None,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    '''
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    '''
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return low_res_pred, medsam_seg


# %% load model and image
parser = argparse.ArgumentParser(
    description="run inference on testing set based on MedSAM"
)
parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    default="image.png",
    help="path to the data folder",
)
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    default="assets/",
    help="path to the segmentation folder",
)
parser.add_argument(
    "--box",
    type=list,
    default=[95, 255, 190, 350],
    help="bounding box of the segmentation target",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="work_dir/MedSAM/medsam_vit_b.pth",
    help="path to the trained model",
)
args = parser.parse_args()

device = args.device
medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
med_sam_checkpoint = "medsam_model_best.pth"
state_dict = torch.load(med_sam_checkpoint, map_location=torch.device('cpu'))
medsam_model.load_state_dict(state_dict['model'])
medsam_model = medsam_model.to(device)

medsam_model.eval()

img_np = io.imread(args.data_path)

if len(img_np.shape) == 2:
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
else:
    img_3c = img_np
img_3c = img_3c[ :, :,:3]
H, W, _ = img_3c.shape
# %% image preprocessing
img_1024 = transform.resize(
    img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
).astype(np.uint8)
img_1024 = (img_1024 - img_1024.min()) / np.clip(
    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
)  # normalize to [0, 1], (H, W, 3)


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
array_size = 1024

# Define the size of your grid
grid_size = 10

# Generate the grid points
x = np.linspace(0, array_size - 1, grid_size)
y = np.linspace(0, array_size - 1, grid_size)

# Generate a grid of coordinates
xv, yv = np.meshgrid(x, y)

# Convert the numpy arrays to lists
xv_list = xv.tolist()
yv_list = yv.tolist()

# Combine the x and y coordinates into a list of list of lists
input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]

# We need to reshape our nxn grid to the expected shape of the input_points tensor
# (batch_size, point_batch_size, num_points_per_image, 2),
# where the last dimension of 2 represents the x and y coordinates of each point.
# batch_size: The number of images you're processing at once.
# point_batch_size: The number of point sets you have for each image.
# num_points_per_image: The number of points in each set.
input_points = torch.tensor(input_points).view(1, grid_size * grid_size, 2)
labels = torch.tensor(np.ones((1, grid_size * grid_size)))  # all labelled as foregrounds

#for i in range(grid_size*grid_size):
#    if img_1024[input_points[0][i][0], input_points[0][i][1]].sum() < 3.0:  # foreground point
#        labels[0][i] = 1

input_points = input_points.to(device)
labels = labels.to(device)

# convert the shape to (3, H, W)
img_1024_tensor = (
    torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
)

box_np = np.array([args.box])
# transfer box_np t0 1024x1024 scale
box_1024 = box_np / np.array([W, H, W, H]) * 1024
with torch.no_grad():
    image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

prob_map, medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W, (input_points, labels))
io.imsave(
    join(args.seg_path, "seg_" + os.path.basename(args.data_path)),
    medsam_seg,
    check_contrast=False,
)

# %% visualize results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img_1024)
#show_box(box_np[0], ax[0])
ax[0].set_title("Input Image")
ax[1].imshow(medsam_seg, cmap='gray')
#show_mask(medsam_seg, ax[1])
#show_box(box_np[0], ax[1])
ax[1].set_title("MedSAM Segmentation")
ax[2].imshow(prob_map)  # Assuming the second image is grayscale
ax[2].set_title("Probability Map")
plt.savefig("medsam_inference.png")
