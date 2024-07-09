import numpy as np
from matplotlib import pyplot as plt

for i in range(0,3):
    mask = np.load(f'data/npy/hedm_powder/gts/img_{i}.npy')
    img = np.load(f'data/npy/hedm_powder/imgs/img_{i}.npy')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))

    ax1.imshow(img)
    ax2.imshow(mask)

    plt.savefig(f'plot_{i}.png')
    plt.show()