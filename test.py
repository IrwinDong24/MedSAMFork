import numpy as np

img_path = "/homes/xma/MedSAM/data/npy/hedm_powder/imgs/img_0.npy"
maks_path = "/homes/xma/MedSAM/data/npy/hedm_powder/gts/img_0.npy"

img = np.load(img_path)
gt = np.load(maks_path)

print(img.shape)
print(gt.shape)
