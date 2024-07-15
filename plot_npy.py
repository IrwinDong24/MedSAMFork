import numpy as np
from matplotlib import pyplot as plt

for i in range(0,1):
    mask = np.load(f'test/npy/hedm_powder/gts/img_{i}.npy')
    img = np.load(f'test/npy/hedm_powder/imgs/img_{i}.npy')

    img = (img - img.min()) / np.clip(
        img.max() - img.min(), a_min=1e-8, a_max=None)
    cmap = plt.cm.viridis
    #img_3c = np.repeat(img[:, :, None], 3, axis=-1)
    img_3c = cmap(img)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))

    ax1.imshow(img_3c)
    ax2.imshow(mask)
    plt.imsave(f'aps_img_{i}.png', img_3c)
    plt.imsave(f'aps_mask_{i}.png', mask)

    #plt.savefig(f'plot_{i}.png')
    plt.show()