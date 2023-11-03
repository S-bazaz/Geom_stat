from skimage import io
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np

import histomicstk as htk

img = io.imread("5.tiff")[:, :, :3]
# plt.imshow(img, cmap = "gray")
# plt.axis("off")
# plt.show()

stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

stains = ["hematoxylin", "eosin","null"]

W = np.array([stain_color_map[st] for st in stains]).T

img = htk.preprocessing.color_deconvolution.color_deconvolution(img, W).Stains[:,:,0]

def tau_activation(img,tau):
    img = np.int32(img>tau)
    return img

img = tau_activation(img, 100)

plt.imshow(img)
plt.axis("off")
plt.show()